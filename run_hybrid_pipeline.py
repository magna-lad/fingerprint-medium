import numpy as np
import xgboost as xgb
import torch
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Imports from your existing files
from graph_minutiae import GraphMinutiae
from load_save import load_users_dictionary
from xgboost_feature_extractor import create_feature_vector_for_pair
from cnn_model import TinyCNN, FingerprintTextureDataset, ContrastiveLoss

# ==========================================
# 1. XGBoost Helper Functions (Reused)
# ==========================================
def prepare_xgb_data(pairs, is_training=False):
    X, y = [], []
    # We do NOT use graph augmentation here because the CNN handles the robustness
    # This keeps the XGBoost part fast and focused on geometry
    for g1, g2, label in tqdm(pairs, desc="XGBoost Feature Extract"):
        X.append(create_feature_vector_for_pair(g1, g2))
        y.append(label)
    return np.array(X), np.array(y)

# ==========================================
# 2. Main Hybrid Pipeline
# ==========================================
def run_hybrid_system():
    print("="*60); print(" HYBRID FINGERPRINT PIPELINE (GRAPH + CNN) "); print("="*60)
    
    # --- A. DATA LOADING & SPLITTING ---
    print("\n[A] Loading Data...")
    users = load_users_dictionary('processed_data.pkl', True)
    analyzer = GraphMinutiae(users)
    analyzer.graph_maker()
    
    # Using 3 impostors per genuine to keep dataset size manageable for Hybrid training
    all_pairs = analyzer.create_graph_pairs(num_impostors_per_genuine=3) 
    train_users, _, test_users = analyzer.get_user_splits()
    train_pairs, _, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_users, [], test_users)

    # --- B. TRAIN XGBOOST (Geometry Expert) ---
    print("\n[B] Training XGBoost (Geometry)...")
    X_train_xgb, y_train_xgb = prepare_xgb_data(train_pairs, is_training=True)
    X_test_xgb, y_test_xgb = prepare_xgb_data(test_pairs, is_training=False)
    
    scaler = StandardScaler()
    X_train_xgb = scaler.fit_transform(X_train_xgb)
    X_test_xgb = scaler.transform(X_test_xgb)
    
    # Standard robust params
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, 
        max_depth=4, 
        learning_rate=0.05, 
        n_jobs=-1, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)
    
    print("Getting XGBoost predictions...")
    xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]
    
    # --- C. TRAIN CNN (Texture Expert) ---
    print("\n[C] Training TinyCNN (Texture) with Augmentation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset with Augmentation ON for Training
    train_dataset = FingerprintTextureDataset(train_pairs, GraphMinutiae.find_true_core, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
    
    cnn = TinyCNN().to(device)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0005) # Slower LR for stability
    
    # 2. Training Loop
    cnn.train()
    epochs = 10 # Increased epochs because augmentation makes the task harder
    for epoch in range(epochs):
        epoch_loss = 0
        for img1, img2, label in tqdm(train_loader, desc=f"CNN Epoch {epoch+1}/{epochs}"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            dist = cnn(img1, img2)
            loss = criterion(dist, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch Loss: {epoch_loss/len(train_loader):.4f}")

    # --- D. HYBRID FUSION ---
    print("\n[D] Performing Hybrid Fusion...")
    cnn.eval()
    
    # 1. Dataset with Augmentation OFF for Testing (Deterministic scoring)
    test_dataset = FingerprintTextureDataset(test_pairs, GraphMinutiae.find_true_core, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    cnn_dists = []
    with torch.no_grad():
        for img1, img2, label in tqdm(test_loader, desc="CNN Inference"):
            img1, img2 = img1.to(device), img2.to(device)
            dist = cnn(img1, img2)
            cnn_dists.extend(dist.cpu().numpy())
    
    cnn_dists = np.array(cnn_dists)
    
    # Convert distances to probabilities
    # Assuming margin is 2.0, Dist=0 is 100% match, Dist=2 is 0% match.
    cnn_probs = 1.0 - (cnn_dists / 2.0) 
    cnn_probs = np.clip(cnn_probs, 0.0, 1.0)
    
    # --- E. FINAL EVALUATION ---
    
    # Fusion Formula: Simple Average
    final_probs = (xgb_probs + cnn_probs) / 2.0
    
    # Metrics
    fpr, tpr, _ = roc_curve(y_test_xgb, final_probs)
    auc_score = auc(fpr, tpr)
    
    print(f"\nFinal Hybrid AUC: {auc_score:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'Hybrid System (AUC={auc_score:.4f})', linewidth=3, color='purple')
    
    # Plot Individual curves
    fpr_x, tpr_x, _ = roc_curve(y_test_xgb, xgb_probs)
    plt.plot(fpr_x, tpr_x, label=f'XGBoost Only (AUC={auc(fpr_x, tpr_x):.4f})', linestyle='--')
    
    fpr_c, tpr_c, _ = roc_curve(y_test_xgb, cnn_probs)
    plt.plot(fpr_c, tpr_c, label=f'CNN Only (AUC={auc(fpr_c, tpr_c):.4f})', linestyle='--')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('Hybrid Fingerprint Matching Results (Augmented CNN)')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_hybrid_system()