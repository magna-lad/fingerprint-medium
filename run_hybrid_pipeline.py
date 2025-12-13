import numpy as np
import xgboost as xgb
import torch
import matplotlib.pyplot as plt
import joblib
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

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
    print("="*60); print(" HYBRID PIPELINE (ROBUST RE-TRY) "); print("="*60)
    
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
    
    # --- C. TRAIN CNN (Balanced Sampling) ---
    print("\n[C] Training CNN (Center-Mass Crop + Balanced Batch)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset
    train_dataset = FingerprintTextureDataset(train_pairs, GraphMinutiae.find_true_core, augment=True)
    val_dataset = FingerprintTextureDataset(val_pairs, GraphMinutiae.find_true_core, augment=False)
    
    # 2. CALCULATE SAMPLER WEIGHTS (50/50 Balance)
    y_train = [label for _, _, label in train_pairs]
    class_counts = np.bincount(y_train)
    weights = 1. / class_counts
    samples_weights = torch.from_numpy(np.array([weights[t] for t in y_train])).double()
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    # 3. Loaders
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    cnn = DeeperCNN().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Optimizer (Start conservative)
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=0.0001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    stopper = EarlyStopping(patience=8, path='best_cnn.pth')
    
    # --- CRITICAL: DEBUG DATA BEFORE TRAIN ---
    print("\n--- DEBUG: Checking Data Integrity ---")
    data_iter = iter(train_loader)
    d_img1, d_img2, d_lbl = next(data_iter)
    
    print(f"Batch Shape: {d_img1.shape}")
    print(f"Batch Mean: {d_img1.mean():.4f}, Max: {d_img1.max():.4f}")
    if d_img1.max() == 0:
        print("ERROR: INPUT IMAGES ARE ALL BLACK! Check Preprocessing.")
        return # Stop execution
    else:
        print("Images contain data. Starting training...\n")

    cnn.train()
    epochs = 50
    
    for epoch in range(epochs):
        # --- TRAINING ---
        cnn.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for img1, img2, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            logits = cnn(img1, img2)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)
        
        avg_train = train_loss / len(train_loader)
        train_acc = correct / total
        
        # --- VALIDATION ---
        cnn.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                logits = cnn(img1, img2)
                loss = criterion(logits, label)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)
        
        avg_val = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        scheduler.step(avg_val)
        
        print(f"   Train Loss: {avg_train:.4f} (Acc: {train_acc:.2%}) | "
              f"Val Loss: {avg_val:.4f} (Acc: {val_acc:.2%}) | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        stopper(avg_val, cnn)
        
        if stopper.early_stop:
            print("Early stopping triggered! Loading best model...")
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
            probs = torch.sigmoid(logits)
            cnn_probs_list.extend(probs.cpu().numpy())
    
    cnn_probs = np.array(cnn_probs_list)
    
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
    plt.title('Hybrid Fingerprint Matching (CenterCrop + Balanced)')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_hybrid_system()