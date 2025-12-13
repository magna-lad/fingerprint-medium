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
from cnn_model import DeeperCNN, FingerprintTextureDataset, EarlyStopping 

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
    
    # Create pairs
    all_pairs = analyzer.create_graph_pairs(num_impostors_per_genuine=3)
    train_users, val_users, test_users = analyzer.get_user_splits()
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
    print("\n[C] Training CNN (BCE Approach)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = FingerprintTextureDataset(train_pairs, GraphMinutiae.find_true_core, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    val_dataset = FingerprintTextureDataset(val_pairs, GraphMinutiae.find_true_core, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    cnn = DeeperCNN().to(device)
    
    # --- CHANGED: Use BCEWithLogitsLoss ---
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=0.0005, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    stopper = EarlyStopping(patience=8, path='best_cnn.pth')
    
    cnn.train()
    epochs = 50
    
    for epoch in range(epochs):
        # --- TRAINING STEP ---
        cnn.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for img1, img2, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            logits = cnn(img1, img2) # Returns logits (unscaled scores)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate rough accuracy for monitoring
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)
        
        avg_train = train_loss / len(train_loader)
        train_acc = correct / total
        
        # --- VALIDATION STEP ---
        cnn.eval()
        val_loss = 0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                logits = cnn(img1, img2)
                loss = criterion(logits, label)
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        
        scheduler.step(avg_val)
        
        print(f"   Train Loss: {avg_train:.4f} (Acc: {train_acc:.2%}) | Val Loss: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        stopper(avg_val, cnn)
        if stopper.early_stop:
            print("Early stopping triggered!")
            break
            
    cnn.load_state_dict(torch.load('best_cnn.pth'))

    # --- D. FUSION ---
    print("\n[D] Performing Hybrid Fusion...")
    cnn.eval()
    
    test_dataset = FingerprintTextureDataset(test_pairs, GraphMinutiae.find_true_core, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    cnn_probs_list = []
    with torch.no_grad():
        for img1, img2, label in tqdm(test_loader, desc="CNN Inference"):
            img1, img2 = img1.to(device), img2.to(device)
            logits = cnn(img1, img2)
            # Apply Sigmoid to get probability [0, 1]
            probs = torch.sigmoid(logits)
            cnn_probs_list.extend(probs.cpu().numpy())
    
    cnn_probs = np.array(cnn_probs_list)
    
    # --- E. EVAL ---
    # Since both XGBoost and CNN now output Prob(Genuine), simple averaging works best
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
    plt.title(f'Hybrid Fingerprint Matching')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_hybrid_system()