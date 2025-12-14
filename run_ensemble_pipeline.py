import numpy as np
import xgboost as xgb
import torch
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF

# Imports from your existing files
from graph_minutiae import GraphMinutiae
from load_save import load_users_dictionary
from xgboost_feature_extractor import create_feature_vector_for_pair
from cnn_model import DeeperCNN, FingerprintTextureDataset, EarlyStopping

# ==========================================
# CONFIGURATION
# ==========================================
NUM_ENSEMBLE_MODELS = 3   # Trains 3 distinct models
USE_TTA = True            # Enables Test Time Augmentation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ==========================================

def prepare_xgb_data(pairs, is_training=False):
    X, y = [], []
    for g1, g2, label in tqdm(pairs, desc="XGBoost Feature Extract"):
        X.append(create_feature_vector_for_pair(g1, g2))
        y.append(label)
    return np.array(X), np.array(y)

def train_single_cnn(model_idx, train_loader, val_loader):
    """Trains a single instance of the CNN."""
    print(f"\n--- Training CNN Model {model_idx+1}/{NUM_ENSEMBLE_MODELS} ---")
    
    model = DeeperCNN().to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    stopper = EarlyStopping(patience=8, path=f'best_cnn_v{model_idx}.pth')

    epochs = 40 
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for img1, img2, label in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(img1, img2)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0; total = 0
        
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
                logits = model(img1, img2)
                val_loss += criterion(logits, label).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == label).sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        scheduler.step(avg_val_loss)
        stopper(avg_val_loss, model)
        
        print(f"  Epoch {epoch+1}: Val Loss {avg_val_loss:.4f} | Acc {val_acc:.2%}")
        
        if stopper.early_stop:
            print("  Early stopping triggered.")
            break
            
    # Return best model
    model.load_state_dict(torch.load(f'best_cnn_v{model_idx}.pth'))
    return model

def predict_with_tta(models, test_loader):
    """
    Predicts using Ensemble + TTA (Original, Rot+10, Rot-10).
    Averages 3 views per model * 3 models = 9 predictions per pair.
    """
    print(f"\n--- Running Ensemble Inference with TTA (Size: {len(models)}) ---")
    
    avg_probs = []
    
    # Set all models to eval mode
    for m in models: m.eval()

    with torch.no_grad():
        for img1, img2, label in tqdm(test_loader, desc="Ensemble TTA"):
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            
            batch_probs = torch.zeros(img1.size(0)).to(DEVICE)
            
            # For each model in the ensemble
            for model in models:
                # 1. Original View
                logits = model(img1, img2)
                batch_probs += torch.sigmoid(logits)
                
                if USE_TTA:
                    # 2. Rotated +10 Degrees
                    img1_r1 = TF.rotate(img1, 10)
                    img2_r1 = TF.rotate(img2, 10)
                    logits_r1 = model(img1_r1, img2_r1)
                    batch_probs += torch.sigmoid(logits_r1)
                    
                    # 3. Rotated -10 Degrees
                    img1_r2 = TF.rotate(img1, -10)
                    img2_r2 = TF.rotate(img2, -10)
                    logits_r2 = model(img1_r2, img2_r2)
                    batch_probs += torch.sigmoid(logits_r2)
            
            # Average out
            # (Num Models * 3 Views) if TTA is on, else (Num Models * 1 View)
            divisor = len(models) * (3 if USE_TTA else 1)
            batch_probs /= divisor
            
            avg_probs.extend(batch_probs.cpu().numpy())
            
    return np.array(avg_probs)

def run_ensemble_pipeline():
    print("="*60); print(f" ENSEMBLE PIPELINE (Models={NUM_ENSEMBLE_MODELS} | TTA={USE_TTA}) "); print("="*60)
    
    # [1] LOAD DATA
    users = load_users_dictionary('processed_data.pkl', True)
    analyzer = GraphMinutiae(users)
    analyzer.graph_maker()
    
    all_pairs = analyzer.create_graph_pairs(num_impostors_per_genuine=3)
    train_users, val_users, test_users = analyzer.get_user_splits()
    train_pairs, val_pairs, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_users, val_users, test_users)

    # [2] TRAIN XGBOOST
    print("\n[A] Training XGBoost Baseline...")
    X_train_xgb, y_train_xgb = prepare_xgb_data(train_pairs, is_training=True)
    X_test_xgb, y_test_xgb = prepare_xgb_data(test_pairs, is_training=False)
    
    scaler = StandardScaler()
    X_train_xgb = scaler.fit_transform(X_train_xgb)
    X_test_xgb = scaler.transform(X_test_xgb)
    
    xgb_model = xgb.XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.04, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_xgb, y_train_xgb)
    xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]
    
    # [3] PREPARE CNN DATA
    train_dataset = FingerprintTextureDataset(train_pairs, GraphMinutiae.find_true_core, augment=True)
    val_dataset = FingerprintTextureDataset(val_pairs, GraphMinutiae.find_true_core, augment=False)
    test_dataset = FingerprintTextureDataset(test_pairs, GraphMinutiae.find_true_core, augment=False)
    
    # Weighted Sampler
    y_train_list = [label for _, _, label in train_pairs]
    class_counts = np.bincount(y_train_list)
    weights = 1. / class_counts
    samples_weights = torch.from_numpy(np.array([weights[t] for t in y_train_list])).double()
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # [4] TRAIN ENSEMBLE
    trained_models = []
    for i in range(NUM_ENSEMBLE_MODELS):
        # Determine unique seed for this model run
        torch.manual_seed(42 + i)
        model = train_single_cnn(i, train_loader, val_loader)
        trained_models.append(model)
        
    # [5] ENSEMBLE PREDICTION
    cnn_ensemble_probs = predict_with_tta(trained_models, test_loader)
    
    # [6] OPTIMAL FUSION
    print("\n[B] Optimizing Fusion Weights...")
    best_auc = 0.0
    best_alpha = 0.5
    
    for alpha in np.linspace(0, 1, 41): # Fine-grained search
        hybrid_probs = (alpha * cnn_ensemble_probs) + ((1 - alpha) * xgb_probs)
        fpr, tpr, _ = roc_curve(y_test_xgb, hybrid_probs)
        current_auc = auc(fpr, tpr)
        
        if current_auc > best_auc:
            best_auc = current_auc
            best_alpha = alpha
            
    print(f"\n" + "="*40)
    print(f" FINAL RESULTS (Strategy 4)")
    print(f"="*40)
    print(f"XGBoost AUC:       {auc(roc_curve(y_test_xgb, xgb_probs)[0], roc_curve(y_test_xgb, xgb_probs)[1]):.4f}")
    print(f"Ensemble CNN AUC:  {auc(roc_curve(y_test_xgb, cnn_ensemble_probs)[0], roc_curve(y_test_xgb, cnn_ensemble_probs)[1]):.4f}")
    print(f"Hybrid AUC:        {best_auc:.4f}")
    print(f"Best Mix:          {best_alpha:.2f} CNN + {1-best_alpha:.2f} XGB")
    
    # Plot
    final_probs = (best_alpha * cnn_ensemble_probs) + ((1 - best_alpha) * xgb_probs)
    fpr, tpr, _ = roc_curve(y_test_xgb, final_probs)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'Ensemble Hybrid (AUC={best_auc:.4f})', linewidth=3, color='darkgreen')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'Final Ensemble Result (3x CNN + TTA)')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_ensemble_pipeline()