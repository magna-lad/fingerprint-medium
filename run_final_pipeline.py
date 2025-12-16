import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

# IMPORTS (Ensure these files are in the same directory)
from graph_minutiae import GraphMinutiae
from load_save import load_users_dictionary
from xgboost_feature_extractor import create_feature_vector_for_pair
from cnn_model import DeeperCNN, FingerprintTextureDataset, EarlyStopping

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_FILE = 'processed_data.pkl' # Output from main.py

def prepare_xgb_features(pairs, desc):
    """Extracts geometric graph features for XGBoost."""
    X, y = [], []
    for g1, g2, label in tqdm(pairs, desc=desc):
        X.append(create_feature_vector_for_pair(g1, g2))
        y.append(label)
    return np.array(X), np.array(y)

def run_hybrid_system():
    print("="*60); print(" PROFESSIONAL HYBRID PIPELINE "); print("="*60)
    print(f"Device: {DEVICE}")
    
    # [1] LOAD DATA AND BUILD GRAPHS
    print("\n[1] Preparing Data...")
    # This automatically looks in 'biometric_cache' folder
    users = load_users_dictionary(DATA_FILE, True) 
    
    if users is None:
        print("ERROR: processed_data.pkl not found. Please run main.py first.")
        return

    # Build Graph Objects
    analyzer = GraphMinutiae(users)
    analyzer.graph_maker()
    
    # Create all possible pairs
    all_pairs = analyzer.create_graph_pairs(num_impostors_per_genuine=3)
    
    # [CRITICAL STEP] USER-DISJOINT SPLIT
    # We use 15% for Validation to tune the fusion weight, 15% for Final Test
    train_u, val_u, test_u = analyzer.get_user_splits(train_ratio=0.70, val_ratio=0.15) 
    
    # Filter pairs so no user leaks between sets
    train_pairs, val_pairs, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_u, val_u, test_u)
    
    if len(val_pairs) == 0:
        print("CRITICAL ERROR: Validation set is empty! Adjust your user split ratios or check dataset size.")
        return

    # [2] TRAIN XGBOOST (GEOMETRY MODEL)
    print("\n[2] Training XGBoost (Geometry Branch)...")
    X_train, y_train = prepare_xgb_features(train_pairs, "Extract Train")
    X_val, y_val = prepare_xgb_features(val_pairs, "Extract Val")
    X_test, y_test = prepare_xgb_features(test_pairs, "Extract Test")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=600, 
        max_depth=6, 
        learning_rate=0.02, 
        eval_metric='logloss', 
        early_stopping_rounds=50, 
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Get Probability Scores
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # [3] TRAIN CNN (TEXTURE MODEL)
    print("\n[3] Training CNN (Texture Branch)...")
    
    # Create PyTorch Datasets
    # augment=True for training to prevent overfitting
    train_ds = FingerprintTextureDataset(train_pairs, GraphMinutiae.find_true_core, augment=True)
    val_ds = FingerprintTextureDataset(val_pairs, GraphMinutiae.find_true_core, augment=False)
    test_ds = FingerprintTextureDataset(test_pairs, GraphMinutiae.find_true_core, augment=False)
    
    # Handle Class Imbalance using WeightedRandomSampler
    # Genuine pairs are rare; we boost their sampling frequency
    class_counts = np.bincount(y_train)
    weights = 1. / class_counts
    sample_weights = [weights[int(t)] for t in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model components
    model = DeeperCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Use OneCycleLR for faster convergence ("Super-Convergence")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    stopper = EarlyStopping(patience=10, path='best_cnn.pth')
    
    print("  > Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(img1, img2)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            t_loss += loss.item()
            
        # Validation Step
        model.eval()
        v_loss = 0
        preds = []
        labels_list = []
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
                logits = model(img1, img2)
                v_loss += criterion(logits, label).item()
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                labels_list.extend(label.cpu().numpy())
        
        avg_v_loss = v_loss / len(val_loader)
        v_auc = auc(roc_curve(labels_list, preds)[0], roc_curve(labels_list, preds)[1])
        
        stopper(avg_v_loss, model)
        print(f"  Ep {epoch+1}: Val Loss {avg_v_loss:.4f} | Val AUC {v_auc:.4f}")
        
        if stopper.early_stop:
            print("  > Early Stopping Triggered.")
            break
            
    # Load best checkpoint before inference
    model.load_state_dict(torch.load('best_cnn.pth'))
    model.eval()
    
    # [4] INFERENCE
    def get_probs(loader, description):
        probs = []
        with torch.no_grad():
            for i1, i2, _ in tqdm(loader, desc=description):
                logits = model(i1.to(DEVICE), i2.to(DEVICE))
                probs.extend(torch.sigmoid(logits).cpu().numpy())
        return np.array(probs)

    print("\n[4] Running Inference...")
    cnn_val_probs = get_probs(val_loader, "CNN Validation")
    cnn_test_probs = get_probs(test_loader, "CNN Test")
    
    # [5] OPTIMIZE FUSION WEIGHTS (SCIENTIFICALLY VALID)
    print("\n[5] Optimizing Fusion Weights on VALIDATION Set...")
    best_val_auc = 0
    best_alpha = 0.5
    
    # Grid search: Try alpha from 0.0 (XGB Only) to 1.0 (CNN Only)
    # We find the alpha that works best on VALIDATION data.
    # We do NOT touch Test data here.
    for alpha in np.linspace(0, 1, 101):
        mix = (alpha * cnn_val_probs) + ((1 - alpha) * xgb_val_probs)
        fpr, tpr, _ = roc_curve(y_val, mix)
        sc = auc(fpr, tpr)
        if sc > best_val_auc:
            best_val_auc = sc
            best_alpha = alpha
            
    print(f"  > Optimal Alpha found: {best_alpha:.2f} (CNN Weight)")
    print(f"  > Best Validation AUC: {best_val_auc:.4f}")
    
    # [6] FINAL EVALUATION ON TEST SET
    print("\n[6] FINAL EVALUATION (TEST SET)")
    
    # Apply the alpha found on validation to the test data
    final_probs = (best_alpha * cnn_test_probs) + ((1 - best_alpha) * xgb_test_probs)
    fpr, tpr, _ = roc_curve(y_test, final_probs)
    test_auc = auc(fpr, tpr)
    
    print("-" * 40)
    print(f"XGBoost Test AUC: {auc(roc_curve(y_test, xgb_test_probs)[0], roc_curve(y_test, xgb_test_probs)[1]):.4f}")
    print(f"CNN Test AUC:     {auc(roc_curve(y_test, cnn_test_probs)[0], roc_curve(y_test, cnn_test_probs)[1]):.4f}")
    print(f"HYBRID TEST AUC:  {test_auc:.4f}")
    print("-" * 40)
    
    # Plotting
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, label=f'Hybrid Model (AUC={test_auc:.4f})', color='purple', linewidth=3)
    plt.plot([0,1],[0,1],'k--')
    plt.title(f"Final Result (Mix: {best_alpha:.2f} CNN / {1-best_alpha:.2f} XGB)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('pro_results.png')
    print("ROC Graph saved to 'pro_results.png'")

if __name__ == '__main__':
    run_hybrid_system()