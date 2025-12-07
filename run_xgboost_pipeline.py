import numpy as np
import xgboost as xgb
import lightgbm as lgb # Import LightGBM as an alternative
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import RandomizedSearchCV # Using RandomizedSearch
from scipy.stats import uniform, randint

from graph_minutiae import GraphMinutiae
from load_save import load_users_dictionary
from xgboost_feature_extractor import create_feature_vector_for_pair

def prepare_data_with_augmentation(pairs, is_training=False, aug_multiplier=3):
    """
    Converts graph pairs into a feature matrix and label vector.
    Applies augmentation to the training set.
    """
    X, y = [], []
    
    # Encapsulate GraphMinutiae's static method for reuse
    builder_instance = GraphMinutiae({})

    for g1, g2, label in tqdm(pairs, desc="Extracting XGBoost Features"):
        # Always add the original pair
        feature_vector = create_feature_vector_for_pair(g1, g2)
        X.append(feature_vector)
        y.append(label)

        # If it's a genuine pair in the training set, add augmented versions
        if is_training and label == 1:
            for _ in range(aug_multiplier):
                # Augment the raw minutiae and build a new graph on the fly
                augmented_minutiae1 = builder_instance.augment_minutiae(g1.raw_minutiae)
                g1_augmented = builder_instance._build_single_graph(augmented_minutiae1,g1.orientation_map, g1.graph_id + "_aug")

                if g1_augmented is not None:
                    aug_feature_vector = create_feature_vector_for_pair(g1_augmented, g2)
                    X.append(aug_feature_vector)
                    y.append(label)

    return np.array(X), np.array(y)


def evaluate_xgboost_model(model, X_test, y_test, test_pairs):
    """
    Evaluates the model, plots results, and performs error analysis.
    """
    print("\n--- Evaluating on Test Set ---")
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred_binary = (y_scores >= 0.5).astype(int) # Use a 0.5 threshold for accuracy calculation

    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Test Accuracy (at 0.5 threshold): {accuracy:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
    eer_threshold = thresholds[eer_index]
    auc_score = auc(fpr, tpr)
    print(f"\nRESULTS -> AUC: {auc_score:.4f} | EER: {eer:.4f} ({eer*100:.2f}%) | EER Threshold: {eer_threshold:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.scatter(fpr[eer_index], tpr[eer_index], c='r', s=100, zorder=5, label=f'EER Point (EER = {eer:.4f})')
    plt.xlabel('False Positive Rate (FPR)'); plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve'); plt.legend(); plt.grid(True); plt.show()

    print("\nDisplaying Feature Importance...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Universal way to plot importance for both XGBoost and LightGBM
    if hasattr(model, 'feature_importances_'):
        xgb.plot_importance(model, ax=ax, max_num_features=20, height=0.5, importance_type='gain')
        plt.title("Model Feature Importance (Gain)"); plt.tight_layout(); plt.show()

    # --- ERROR ANALYSIS ---
    print("\n--- Performing Error Analysis ---")
    false_negatives = [{'score': y_scores[i], 'g1_id': test_pairs[i][0].graph_id, 'g2_id': test_pairs[i][1].graph_id}
                       for i in range(len(y_test)) if y_test[i] == 1 and y_pred_binary[i] == 0]
    false_positives = [{'score': y_scores[i], 'g1_id': test_pairs[i][0].graph_id, 'g2_id': test_pairs[i][1].graph_id}
                       for i in range(len(y_test)) if y_test[i] == 0 and y_pred_binary[i] == 1]
    false_negatives.sort(key=lambda x: x['score'])
    false_positives.sort(key=lambda x: x['score'], reverse=True)
    print(f"\nFound {len(false_negatives)} False Negatives (genuine pairs missed). Top 5 worst misses:")
    for item in false_negatives[:5]: print(f"  - Predicted Score: {item['score']:.4f} | Pair: {item['g1_id']} vs {item['g2_id']}")
    print(f"\nFound {len(false_positives)} False Positives (impostors accepted). Top 5 worst mistakes:")
    for item in false_positives[:5]: print(f"  - Predicted Score: {item['score']:.4f} | Pair: {item['g1_id']} vs {item['g2_id']}")


def run_xgboost_fingerprint_pipeline():
    print("="*60); print(" FINGERPRINT ML PIPELINE (TRAINING & SAVING) "); print("="*60)

    #### start 

    print("\n[1/5] Loading data...") 
    users = load_users_dictionary('processed_data.pkl', True)
    analyzer = GraphMinutiae(users)


    print("\n[2/5] Building graphs...")
    analyzer.graph_maker()


    print("\n[3/5] Creating pairs for evaluation..."); all_pairs = analyzer.create_graph_pairs(num_impostors_per_genuine=4)
    print("\n[4/5] Splitting users for disjoint sets...")
    train_users, _, test_users = analyzer.get_user_splits()
    train_pairs, _, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_users, [], test_users)

    print("\n[5/5] Preparing data with augmentation for training...")
    X_train, y_train = prepare_data_with_augmentation(train_pairs, is_training=True,aug_multiplier=7)
    X_test, y_test = prepare_data_with_augmentation(test_pairs, is_training=False) # NO augmentation on test set

    # --- Calculate scale_pos_weight for handling class imbalance ---
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_weight = num_neg / num_pos
    print(f"\nTraining set contains {num_pos} positive samples and {num_neg} negative samples.")
    print(f"Using scale_pos_weight: {scale_weight:.2f}")

    # --- Hyperparameter Tuning with RandomizedSearchCV ---
    print(f"\n--- Starting Hyperparameter Tuning on {X_train.shape[0]} training samples ---")
    param_dist = {
        'n_estimators': randint(200, 1000),
        'max_depth': randint(6, 15),
        'learning_rate': uniform(0.01, 0.15),
        'subsample': uniform(0.6, 0.4), # 0.6 to 1.0
        'colsample_bytree': uniform(0.3, 0.5), # 0.6 to 1.0
        'gamma': [0, 0.1, 0.25, 0.5, 1.0, 2.0],
        'reg_lambda': uniform(1, 15) # L2 regularization
    }
    
    # --- CHOOSE YOUR MODEL ---
    # Option 1: XGBoost
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_jobs=-1, scale_pos_weight=scale_weight, use_label_encoder=False)
    
    # Option 2: LightGBM (often faster and sometimes more accurate)
    # model = lgb.LGBMClassifier(objective='binary', metric='logloss', n_jobs=-1, scale_pos_weight=scale_weight)
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings that are sampled
        scoring='roc_auc',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best cross-validation AUC score: {random_search.best_score_:.4f}")
    best_model = random_search.best_estimator_

    # --- SAVE THE FINAL MODEL ---
    model_filename = 'fingerprint_verifier.joblib'
    print(f"\nSaving the best model to '{model_filename}'...")
    joblib.dump(best_model, model_filename)
    print("Model saved.")

    # Evaluate the final, tuned model
    evaluate_xgboost_model(best_model, X_test, y_test, test_pairs)
    print("\n--- XGBOOST PIPELINE COMPLETE ---")

if __name__ == '__main__':
    run_xgboost_fingerprint_pipeline()