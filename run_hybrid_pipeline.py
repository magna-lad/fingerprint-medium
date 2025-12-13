import numpy as np
import xgboost as xgb
import torch
import matplotlib.pyplot as plt
import joblib
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Imports
from graph_minutiae import GraphMinutiae
from load_save import load_users_dictionary
from xgboost_feature_extractor import create_feature_vector_for_pair
from cnn_model import DeeperCNN, FingerprintTextureDataset, ContrastiveLoss, EarlyStopping

def prepare_xgb_data(pairs, is_training=False):
    X, y = [], []
    for g1, g2, label in tqdm(pairs, desc="XGBoost Feature Extract"):
        X.append(create_feature_vector_for_pair(g1, g2))
        y.append(label)
    return np.array(X), np.array(y)

def run_hybrid_system():
    print("="*60); print(" HYBRID PIPELINE (GRAPH + CNN + EARLY STOPPING) "); print("="*60)
    
    # --- A. LOADING ---
    print("\n[A] Loading Data...")
    users = load_users_dictionary('/kaggle/input/processed-data/processed_data.pkl', True)
    analyzer = GraphMinutiae(users)
    analyzer.graph_maker()
    
    all_pairs = analyzer.create_graph_pairs(num_impostors_per_genuine=3)
    train_users, val_users, test_users = analyzer.get_user_splits()
    
    # !!! KEY CHANGE: We now actively capture val_pairs !!!
    train_pairs, val_pairs, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_users, val_users, test_users)

    # --- B. XGBOOST ---
    print("\n[B] Training XGBoost (Geometry)...")
    X_train_xgb, y_train_xgb = prepare_xgb_data(train_pairs, is_training=True)
    X_test_xgb, y_test_xgb = prepare_xgb_data(test_pairs, is_training=False)
    
    scaler = StandardScaler()
    X_train_xgb = scaler.fit_transform(X_train_xgb)
    X_test_xgb = scaler.transform(X_test_xgb)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, 
        n_jobs=-1, use_label_encoder=False, eval_metric='logloss'
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)
    xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]
    
    # --- C. TRAIN CNN (Texture) ---
    print("\n[C] Training CNN with Early Stopping...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Data Loaders
    # Train: Augmentation ON
    train_dataset = FingerprintTextureDataset(train_pairs, GraphMinutiae.find_true_core, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Val: Augmentation OFF (We measure performance on clean data)
    val_dataset = FingerprintTextureDataset(val_pairs, GraphMinutiae.find_true_core, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    cnn = DeeperCNN().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)
    
    # Initialize Early Stopping
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    stopper = EarlyStopping(patience=5, path='best_cnn.pth')
    
    # 2. Training Loop (Up to 50 Epochs)
    cnn.train()
    epochs = 50
    
    for epoch in range(epochs):
        # --- TRAINING STEP ---
        cnn.train()
        train_loss = 0
        for img1, img2, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            dist = cnn(img1, img2)
            loss = criterion(dist, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # --- VALIDATION STEP ---
        cnn.eval()
        val_loss = 0
        with torch.no_grad():
            for img1, img2, label in val_loader: # No tqdm to keep log clean
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                dist = cnn(img1, img2)
                loss = criterion(dist, label)
                val_loss += loss.item()
        
        scheduler.step()

        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(val_loader)
        
        print(f"   Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # --- EARLY STOPPING CHECK ---
        stopper(avg_val, cnn)
        
        if stopper.early_stop:
            print("Early stopping triggered! Loading best model...")
            break
            
    # Load the best weights saved by EarlyStopping
    cnn.load_state_dict(torch.load('best_cnn.pth'))

    # --- D. FUSION ---
    print("\n[D] Performing Hybrid Fusion...")
    cnn.eval()
    
    # Test Data Loader (No Augmentation)
    test_dataset = FingerprintTextureDataset(test_pairs, GraphMinutiae.find_true_core, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    cnn_dists = []
    with torch.no_grad():
        for img1, img2, label in tqdm(test_loader, desc="CNN Inference"):
            img1, img2 = img1.to(device), img2.to(device)
            dist = cnn(img1, img2)
            cnn_dists.extend(dist.cpu().numpy())
    
    cnn_dists = np.array(cnn_dists)
    
    # Normalize probabilities
    cnn_probs = 1.0 - (cnn_dists / 2.0) 
    cnn_probs = np.clip(cnn_probs, 0.0, 1.0)
    
    # --- E. EVAL ---
    final_probs = (xgb_probs + cnn_probs) / 2.0
    
    fpr, tpr, _ = roc_curve(y_test_xgb, final_probs)
    auc_score = auc(fpr, tpr)
    
    print(f"\nFinal Hybrid AUC: {auc_score:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'Hybrid System (AUC={auc_score:.4f})', linewidth=3, color='purple')
    
    fpr_x, tpr_x, _ = roc_curve(y_test_xgb, xgb_probs)
    plt.plot(fpr_x, tpr_x, label=f'XGBoost (AUC={auc(fpr_x, tpr_x):.4f})', linestyle='--')
    
    fpr_c, tpr_c, _ = roc_curve(y_test_xgb, cnn_probs)
    plt.plot(fpr_c, tpr_c, label=f'CNN (AUC={auc(fpr_c, tpr_c):.4f})', linestyle='--')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('Hybrid Fingerprint Matching (50 Epochs + Early Stop)')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_hybrid_system()