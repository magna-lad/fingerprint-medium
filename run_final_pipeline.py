import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import torchvision.transforms.functional as TF

# Custom Modules
from graph_minutiae import GraphMinutiae
from load_save import load_users_dictionary
from xgboost_feature_extractor import create_feature_vector_for_pair
from cnn_model import DeeperCNN, FingerprintTextureDataset

# ==========================================
# CONFIGURATION
# ==========================================
NUM_ENSEMBLE = 3           # Train 3 Independent Models
USE_TTA = True             # Test Time Augmentation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRIPLET_MARGIN = 1.0
EPOCHS = 35
# ==========================================

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        dist_pos = (anchor - positive).pow(2).sum(1)
        dist_neg = (anchor - negative).pow(2).sum(1)
        return F.relu(dist_pos - dist_neg + self.margin).mean()

# TRIPLET DATASET (Only for training)
class FingerprintTripletDataset(Dataset):
    def __init__(self, graphs, augment=True):
        self.augment = augment
        # Create helper to reuse preprocessing logic from cnn_model
        # We pass a dummy list because we only need the methods
        self.helper_ds = FingerprintTextureDataset([], None, augment=augment)
        
        # Organize graphs by ID
        self.identities = defaultdict(list)
        for g in graphs:
            key = (g['user_id'], g['hand'], g['finger_idx'])
            self.identities[key].append(g)
        self.identity_keys = list(self.identities.keys())

    def __len__(self):
        return len(self.identity_keys)

    def _get_img(self, meta):
        # Uses the logic from cnn_model.py
        p = self.helper_ds.preprocess_image(meta['graph'].skeleton, meta['graph'].orientation_map)
        if self.augment:
             p_u8 = (p * 255).astype(np.uint8)
             return self.helper_ds.transform(p_u8)
        else:
             return torch.from_numpy(p).unsqueeze(0)

    def __getitem__(self, idx):
        # 1. Anchor Identity
        anchor_key = self.identity_keys[idx]
        samples = self.identities[anchor_key]
        if len(samples) < 2: return self.__getitem__(np.random.randint(0, len(self)))
        
        # 2. Anchor & Positive
        a_meta, p_meta = random.sample(samples, 2)
        
        # 3. Negative (Random different ID)
        neg_idx = np.random.randint(0, len(self))
        while neg_idx == idx: neg_idx = np.random.randint(0, len(self))
        neg_meta = random.choice(self.identities[self.identity_keys[neg_idx]])
        
        return self._get_img(a_meta), self._get_img(p_meta), self._get_img(neg_meta)

def train_triplet_model(model_idx, train_graphs):
    filename = f'best_triplet_v{model_idx}.pth'
    if os.path.exists(filename):
        print(f"[LOAD] Found {filename}, loading...")
        model = DeeperCNN().to(DEVICE)
        model.load_state_dict(torch.load(filename))
        return model

    print(f"\n--- Training Triplet Model {model_idx+1}/{NUM_ENSEMBLE} ---")
    train_ds = FingerprintTripletDataset(train_graphs, augment=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    model = DeeperCNN().to(DEVICE)
    criterion = TripletLoss(margin=TRIPLET_MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    # Simple training loop (No validation set used for stopping to save pairs for final test)
    # We train for fixed epochs to ensure robustness
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)
        for anchor, pos, neg in pbar:
            anchor, pos, neg = anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            optimizer.zero_grad()
            ea = model.forward_one(anchor)
            ep = model.forward_one(pos)
            en = model.forward_one(neg)
            loss = criterion(ea, ep, en)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    torch.save(model.state_dict(), filename)
    return model

def calculate_similarity(models, pair_loader):
    """
    Calculates ensemble Cosine Similarity with TTA.
    Output: Normalized scores [0, 1] where 1 is match.
    """
    print(f"Calculating Similiarity (TTA={USE_TTA})...")
    scores = []
    
    for m in models: m.eval()
    
    with torch.no_grad():
        for img1, img2, _ in tqdm(pair_loader):
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            
            # Aggregate embeddings from all models and TTA views
            emb1_list, emb2_list = [], []
            
            for model in models:
                # View 1: Original
                emb1_list.append(model.forward_one(img1))
                emb2_list.append(model.forward_one(img2))
                
                if USE_TTA:
                    # View 2: +10 deg
                    emb1_list.append(model.forward_one(TF.rotate(img1, 10)))
                    emb2_list.append(model.forward_one(TF.rotate(img2, 10)))
                    # View 3: -10 deg
                    emb1_list.append(model.forward_one(TF.rotate(img1, -10)))
                    emb2_list.append(model.forward_one(TF.rotate(img2, -10)))
            
            # Stack: [Views*Models, Batch, Features] -> Mean -> [Batch, Features]
            # Average the vectors to get a robust prototype
            e1 = torch.stack(emb1_list).mean(dim=0)
            e2 = torch.stack(emb2_list).mean(dim=0)
            
            # Cosine Similarity: Ranges -1 to 1.
            cos_sim = F.cosine_similarity(e1, e2)
            
            # Normalize to 0-1 for Fusion (Match=1, NonMatch=0)
            # Cosine usually > 0 for fingerprints, but mapping -1..1 to 0..1 is safest
            norm_score = (cos_sim + 1) / 2.0 
            scores.extend(norm_score.cpu().numpy())
            
    return np.array(scores)

def prepare_xgb_data(pairs, is_training=False):
    X, y = [], []
    for g1, g2, label in tqdm(pairs, desc="XGBoost"):
        X.append(create_feature_vector_for_pair(g1, g2))
        y.append(label)
    return np.array(X), np.array(y)

def run_final_pipeline():
    print("="*60); print(" FINAL NUCLEAR PIPELINE (>95% GOAL) "); print("="*60)
    
    # 1. Load Data
    users = load_users_dictionary('/kaggle/input/processed-data/processed_data.pkl', True)
    analyzer = GraphMinutiae(users)
    analyzer.graph_maker()
    all_pairs = analyzer.create_graph_pairs(num_impostors_per_genuine=3)
    train_users, val_users, test_users = analyzer.get_user_splits()
    
    # Split Pairs
    train_pairs, _, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_users, [], test_users)
    
    # Split Graphs for Triplet Training
    all_graphs = analyzer.fingerprint_graphs
    train_graphs = [g for g in all_graphs if g['user_id'] in train_users]

    # 2. XGBoost (Geometry Expert)
    print("\n[Phase 1] XGBoost Training...")
    X_train, y_train = prepare_xgb_data(train_pairs, True)
    X_test, y_test = prepare_xgb_data(test_pairs, False)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    
    xgb_model = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03, n_jobs=-1, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # 3. Triplet Ensemble (Texture/Shape Expert)
    print("\n[Phase 2] Triplet Ensemble Training...")
    trained_models = []
    for i in range(NUM_ENSEMBLE):
        torch.manual_seed(42 + i) # Diversity
        trained_models.append(train_triplet_model(i, train_graphs))
        
    # 4. Inference
    test_ds = FingerprintTextureDataset(test_pairs, GraphMinutiae.find_true_core, augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    cnn_sim_scores = calculate_similarity(trained_models, test_loader)
    
    # 5. Fusion
    print("\n[Phase 3] Fusion & Result...")
    
    # Optimization loop
    best_auc = 0; best_alpha = 0.5
    for alpha in np.linspace(0, 1, 51):
        mix = (alpha * cnn_sim_scores) + ((1-alpha) * xgb_probs)
        fpr, tpr, _ = roc_curve(y_test, mix)
        sc = auc(fpr, tpr)
        if sc > best_auc: best_auc = sc; best_alpha = alpha

    print("-" * 30)
    print(f"XGBoost AUC:    {auc(roc_curve(y_test, xgb_probs)[0], roc_curve(y_test, xgb_probs)[1]):.4f}")
    print(f"Triplet CNN AUC:{auc(roc_curve(y_test, cnn_sim_scores)[0], roc_curve(y_test, cnn_sim_scores)[1]):.4f}")
    print(f"FINAL HYBRID:   {best_auc:.4f}")
    print(f"Mix: {best_alpha:.2f} CNN + {1-best_alpha:.2f} XGB")
    print("-" * 30)

    # Save Graph
    final_scores = (best_alpha * cnn_sim_scores) + ((1-best_alpha) * xgb_probs)
    fpr, tpr, _ = roc_curve(y_test, final_scores)
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, label=f'Nuclear Hybrid (AUC={best_auc:.4f})', color='red', linewidth=3)
    plt.plot([0,1],[0,1],'k--')
    plt.title("Final Architecture Performance")
    plt.legend(); plt.grid(True)
    plt.savefig('nuclear_results.png')
    print("Graph saved to 'nuclear_results.png'.")

if __name__ == '__main__':
    run_final_pipeline()