import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler

# Custom modules
from graph_minutiae import GraphMinutiae
from load_save import load_users_dictionary
from xgboost_feature_extractor import create_feature_vector_for_pair

def prepare_data_with_augmentation(pairs, is_training=False, aug_multiplier=3):
    """Converts graph pairs into a feature matrix and label vector."""
    X, y = [], []
    builder_instance = GraphMinutiae({})

    for g1, g2, label in tqdm(pairs, desc="Extracting XGBoost Features"):
        feature_vector = create_feature_vector_for_pair(g1, g2)
        X.append(feature_vector)
        y.append(label)

        if is_training and label == 1:
            for _ in range(aug_multiplier):
                augmented_minutiae1 = builder_instance.augment_minutiae(g1.raw_minutiae)
                g1_augmented = builder_instance._build_single_graph(augmented_minutiae1, g1.orientation_map, g1.graph_id + "_aug")

                if g1_augmented is not None:
                    aug_feature_vector = create_feature_vector_for_pair(g1_augmented, g2)
                    X.append(aug_feature_vector)
                    y.append(label)

    return np.array(X), np.array(y)


def evaluate_xgboost_model(model, X_test, y_test, test_pairs):
    print("\n--- Evaluating on Test Set ---")
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # --- SAVE PREDICTIONS FOR HYBRID CNN FUSION ---
    print("Saving predictions for future hybrid fusion...")
    # Creates a dictionary mapping pair_ID -> score
    predictions = {}
    for i in range(len(test_pairs)):
        pair_key = f"{test_pairs[i][0].graph_id}_VS_{test_pairs[i][1].graph_id}"
        predictions[pair_key] = float(y_scores[i])
    joblib.dump(predictions, 'xgboost_predictions.pkl')
    # ----------------------------------------------

    y_pred_binary = (y_scores >= 0.5).astype(int) 

    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Test Accuracy (at 0.5 threshold): {accuracy:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
    eer_threshold = thresholds[eer_index]
    auc_score = auc(fpr, tpr)
    print(f"\nRESULTS -> AUC: {auc_score:.4f} | EER: {eer:.4f} ({eer*100:.2f}%) | EER Threshold: {eer_threshold:.4f}")

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.scatter(fpr[eer_index], tpr[eer_index], c='r', s=100, zorder=5, label=f'EER Point (EER = {eer:.4f})')
    plt.title('Receiver Operating Characteristic (ROC) Curve'); plt.legend(); plt.grid(True); plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    if hasattr(model, 'feature_importances_'):
        xgb.plot_importance(model, ax=ax, max_num_features=20, height=0.5, importance_type='gain')
        plt.title("Model Feature Importance (Gain)"); plt.tight_layout(); plt.show()

    # --- ERROR ANALYSIS ---
    false_negatives = [{'score': y_scores[i], 'g1_id': test_pairs[i][0].graph_id, 'g2_id': test_pairs[i][1].graph_id}
                       for i in range(len(y_test)) if y_test[i] == 1 and y_pred_binary[i] == 0]
    false_positives = [{'score': y_scores[i], 'g1_id': test_pairs[i][0].graph_id, 'g2_id': test_pairs[i][1].graph_id}
                       for i in range(len(y_test)) if y_test[i] == 0 and y_pred_binary[i] == 1]
    
    false_negatives.sort(key=lambda x: x['score'])
    false_positives.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nFound {len(false_negatives)} False Negatives. Top 5 worst misses:")
    for item in false_negatives[:5]: print(f"  - Score: {item['score']:.4f} | {item['g1_id']} vs {item['g2_id']}")
    print(f"\nFound {len(false_positives)} False Positives. Top 5 worst mistakes:")
    for item in false_positives[:5]: print(f"  - Score: {item['score']:.4f} | {item['g1_id']} vs {item['g2_id']}")


def run_xgboost_fingerprint_pipeline():
    print("="*60); print(" FINGERPRINT ML PIPELINE (TRAINING & SAVING) "); print("="*60)

    print("\n[1/5] Loading data...") 
    users = load_users_dictionary('processed_data.pkl', True)
    analyzer = GraphMinutiae(users)

    print("\n[2/5] Building graphs...")
    analyzer.graph_maker()

    print("\n[3/5] Creating pairs...") 
    all_pairs = analyzer.create_graph_pairs(num_impostors_per_genuine=4)
    train_users, _, test_users = analyzer.get_user_splits()
    train_pairs, _, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_users, [], test_users)

    print("\n[5/5] Preparing data with augmentation...")
    # Augment multiplier 5, using low-jitter/low-dropout settings
    X_train, y_train = prepare_data_with_augmentation(train_pairs, is_training=True, aug_multiplier=5)
    X_test, y_test = prepare_data_with_augmentation(test_pairs, is_training=False)

    print("\nScaling Features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'feature_scaler.joblib')

    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_weight = (num_neg / num_pos) * 2.0 
    print(f"\nUsing boosted scale_pos_weight: {scale_weight:.2f}")

    print(f"\n--- Training Robust XGBoost Model ---")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',      
        n_estimators=450,       
        max_depth=5,            # Slightly deeper to learn combined features
        learning_rate=0.02,     
        colsample_bytree=0.5,   
        subsample=0.7,          
        reg_alpha=1.0,          
        reg_lambda=5.0,         
        n_jobs=-1,
        scale_pos_weight=scale_weight,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    best_model = model
    model_filename = 'fingerprint_verifier.joblib'
    print(f"\nSaving the model to '{model_filename}'...")
    joblib.dump(best_model, model_filename)

    evaluate_xgboost_model(best_model, X_test, y_test, test_pairs)
    print("\n--- XGBOOST PIPELINE COMPLETE ---")

if __name__ == '__main__':
    run_xgboost_fingerprint_pipeline()