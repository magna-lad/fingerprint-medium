import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tq
from torch.nn import Sequential, Linear, LeakyReLU, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve, auc
from graph_minutiae import GraphMinutiae
from torch_geometric.data import Batch

# =================================================================================
# ===== Final Model Architecture and Loss =========================================
# =================================================================================

class FingerprintAngularGNN(torch.nn.Module):
    def __init__(self, node_features=9, hidden_dim=256, embedding_dim=256, num_conv_layers=5, dropout=0.4, heads=6):
        super().__init__()
        self.convs = torch.nn.ModuleList([GATv2Conv(node_features, hidden_dim, heads=heads, concat=True)])
        self.bns = torch.nn.ModuleList([BatchNorm1d(hidden_dim * heads)])
        for _ in range(num_conv_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            self.bns.append(BatchNorm1d(hidden_dim * heads))
        self.dropout = Dropout(p=dropout)
        self.fc1 = Linear(hidden_dim * heads, hidden_dim); self.bn_fc = BatchNorm1d(hidden_dim)
        self.fc2 = Linear(hidden_dim, embedding_dim); self.bn_embedding = BatchNorm1d(embedding_dim)

    def forward_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index); x = self.bns[i](x); x = F.leaky_relu(x); x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.fc1(x); x = self.bn_fc(x); x = F.leaky_relu(x); x = self.dropout(x)
        x = self.fc2(x); x = self.bn_embedding(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, data):
        return self.forward_embedding(data)

class AngularMarginLoss(torch.nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super().__init__()
        self.in_features, self.out_features, self.s, self.m = in_features, out_features, s, m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = np.cos(m), np.sin(m)
        self.th, self.mm = np.cos(np.pi - m), np.sin(np.pi - m) * m
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, embedding, label):
        cosine = F.linear(embedding, F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=embedding.device).scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return self.criterion(output, label)

# =================================================================================
# ===== Datasets and Training/Evaluation Logic (DEFINITIVELY CORRECTED) ===========
# =================================================================================
class AugmentedFingerprintDataset(Dataset):
    """
    A dataset that handles on-the-fly augmentation of a pre-built graph's features.
    This is a much more robust design that avoids graph-building failures.
    """
    def __init__(self, train_meta, raw_data, id_map):
        self.train_meta = train_meta
        self.raw_data = raw_data # We still need this to get the original un-normalized minutiae
        self.id_map = id_map
        # We need an instance of GraphMinutiae to access its helper methods
        self.feature_normalizer = GraphMinutiae(raw_data)

    def __len__(self):
        return len(self.train_meta)

    def __getitem__(self, idx):
        # 1. Get the metadata for the desired fingerprint
        info = self.train_meta[idx]
        uid, hand, finger, impr = info['user_id'], info['hand'], info['finger_idx'], info['impression_idx']
        
        # 2. Get the corresponding pre-built graph
        original_graph = info['graph']
        
        # 3. Create a deep copy to modify without affecting the original object
        augmented_graph = original_graph.clone()

        # 4. Retrieve the ORIGINAL raw minutiae for this graph
        minutiae = self.raw_data[uid]['fingers'][hand][finger][impr]['minutiae']
        
        # 5. Augment the raw minutiae
        augmented_minutiae = self.feature_normalizer.augment_minutiae(minutiae)
        
        # 6. Re-calculate the normalized node features from the augmented minutiae
        # This will create new coordinates, angles, and core-point features
        new_node_features = self.feature_normalizer.normalize_minutiae_features(augmented_minutiae)
        
        # 7. Update the copied graph's node features (`x`)
        augmented_graph.x = torch.from_numpy(new_node_features).float()
        
        # 8. Get the integer label
        label = self.id_map[uid]
        
        return augmented_graph, label

class PairDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def train_model(train_graphs_meta, val_pairs, raw_users_data, device, num_epochs=200, batch_size=64, learning_rate=5e-5):
    user_ids = sorted(list({info['user_id'] for info in train_graphs_meta}))
    user_id_to_int = {uid: i for i, uid in enumerate(user_ids)}
    num_train_users = len(user_ids) # This will be ~350 now
    print(f"Discovered {num_train_users} unique users in the training set.")
    
    model = FingerprintAngularGNN().to(device)
    criterion = AngularMarginLoss(in_features=256, out_features=num_train_users).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_dataset = AugmentedFingerprintDataset(train_graphs_meta, raw_users_data, user_id_to_int)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    history = {'train_loss': [], 'val_genuine_dist': [], 'val_impostor_dist': []}
    
    # --- EARLY STOPPING LOGIC ---
    best_val_gap = -1.0  # Start with a negative gap
    patience_counter = 0
    early_stop_patience = 20 # Stop if the gap doesn't improve for 30 epochs

    print(f"\n--- Starting Final Training (Early Stopping Patience: {early_stop_patience}) ---")
    for epoch in range(num_epochs):
        model.train(); criterion.train()
        total_train_loss = 0.0
        for graphs, labels in tq(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            graphs, labels = graphs.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(graphs)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        model.eval()
        genuine_dists, impostor_dists = [], []
        with torch.no_grad():
            for g1, g2, label in val_pairs:
                emb1 = model.forward_embedding(g1.to(device))
                emb2 = model.forward_embedding(g2.to(device))
                dist = F.pairwise_distance(emb1, emb2).item()
                (genuine_dists if label == 1 else impostor_dists).append(dist)
        avg_genuine, avg_impostor = np.mean(genuine_dists) if genuine_dists else 0, np.mean(impostor_dists) if impostor_dists else 0
        
        # We monitor the GAP, not the loss, for validation performance
        current_val_gap = avg_impostor - avg_genuine
        
        history['train_loss'].append(avg_train_loss)
        history['val_genuine_dist'].append(avg_genuine); history['val_impostor_dist'].append(avg_impostor)
        print(f"Train Loss: {avg_train_loss:.4f} | Genuine: {avg_genuine:.4f} | Impostor: {avg_impostor:.4f} | GAP: {current_val_gap:.4f}")

        # --- CHECK FOR IMPROVEMENT AND SAVE BEST MODEL ---
        if current_val_gap > best_val_gap:
            best_val_gap = current_val_gap
            patience_counter = 0
            print(f"  -> New best validation gap: {best_val_gap:.4f}. Saving model.")
            torch.save(model.state_dict(), 'best_fingerprint_model.pth')
        else:
            patience_counter += 1
            print(f"  -> No improvement in validation gap for {patience_counter} epoch(s).")
            
        if patience_counter >= early_stop_patience:
            print(f"\n--- Early stopping triggered after {epoch + 1} epochs. Best gap: {best_val_gap:.4f} ---")
            break
    
    print("Loading best model for final evaluation.")
    model.load_state_dict(torch.load('best_fingerprint_model.pth'))
    return model, history


def evaluate_model(model, test_pairs, device):
    # This function is correct and needs no changes
    model.eval()
    distances, labels = [], []
    print("\n--- Evaluating on Test Set ---")
    with torch.no_grad():
        for g1, g2, label in tq(test_pairs, desc="Testing"):
            emb1 = model.forward_embedding(g1.to(device)); emb2 = model.forward_embedding(g2.to(device))
            distances.append(F.pairwise_distance(emb1, emb2).item())
            labels.append(label)
    distances, labels = np.array(distances), np.array(labels)
    scores = -distances
    fpr, tpr, thresholds = roc_curve(labels, scores); fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    auc_score, eer = auc(fpr, tpr), (fpr[eer_index] + fnr[eer_index]) / 2.0
    eer_threshold = -thresholds[eer_index]
    print(f"\nRESULTS -> AUC: {auc_score:.4f} | EER: {eer:.4f} ({eer*100:.2f}%) | Threshold: {eer_threshold:.4f}")
    plt.figure(figsize=(8, 8)); plt.plot(fpr, tpr, label=f'ROC (AUC={auc_score:.4f})'); plt.plot([0, 1], [0, 1], 'k--', label='Random'); plt.scatter(fpr[eer_index], tpr[eer_index], c='r', zorder=5, label=f'EER={eer:.4f}')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.grid(True); plt.show()
    plt.figure(figsize=(10, 6)); plt.hist(distances[labels==1], bins=50, alpha=0.7, density=True, label='Genuine'); plt.hist(distances[labels==0], bins=50, alpha=0.7, density=True, label='Impostor')
    plt.axvline(eer_threshold, c='r', ls='--', label=f'Threshold={eer_threshold:.4f}'); plt.title('Distance Distributions'); plt.xlabel('Distance'); plt.ylabel('Density'); plt.legend(); plt.grid(True); plt.show()
    return {'auc': auc_score, 'eer': eer}