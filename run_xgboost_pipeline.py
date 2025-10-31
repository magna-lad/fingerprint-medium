import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV

# --- Local Imports ---
from graph_minutiae import GraphMinutiae
from load_save import load_users_dictionary
from xgboost_feature_extractor import create_feature_vector_for_pair

def prepare_data_for_xgboost(pairs):
    """
    Converts a list of graph pairs into a feature matrix (X) and a label vector (y).
    """
    X, y = [], []
    for g1, g2, label in tqdm(pairs, desc="Extracting XGBoost Features"):
        feature_vector = create_feature_vector_for_pair(g1, g2)
        X.append(feature_vector)
        y.append(label)
    return np.array(X), np.array(y)

def evaluate_xgboost_model(model, X_test, y_test, test_pairs):
    """
    Evaluates the model, plots results, and performs error analysis on the worst predictions.
    """
    print("\n--- Evaluating on Test Set ---")
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred_binary = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    fnr = 1 - tpr
    
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
    eer_threshold = thresholds[eer_index]
    
    auc_score = auc(fpr, tpr)
    
    print(f"\nRESULTS -> AUC: {auc_score:.4f} | EER: {eer:.4f} ({eer*100:.2f}%) | Probability Threshold: {eer_threshold:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.scatter(fpr[eer_index], tpr[eer_index], c='r', s=100, zorder=5, label=f'EER Point (EER = {eer:.4f})')
    plt.xlabel('False Positive Rate (FPR)'); plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve'); plt.legend(); plt.grid(True); plt.show()

    print("\nDisplaying Feature Importance...")
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax, max_num_features=20, height=0.5)
    plt.title("XGBoost Feature Importance"); plt.tight_layout(); plt.show()
    
    # --- ERROR ANALYSIS SECTION ---
    print("\n--- Performing Error Analysis ---")
    
    false_negatives = [{'score': y_scores[i], 'g1_id': test_pairs[i][0].graph_id, 'g2_id': test_pairs[i][1].graph_id}
                       for i in range(len(y_test)) if y_test[i] == 1 and y_scores[i] < 0.5]
                       
    false_positives = [{'score': y_scores[i], 'g1_id': test_pairs[i][0].graph_id, 'g2_id': test_pairs[i][1].graph_id}
                       for i in range(len(y_test)) if y_test[i] == 0 and y_scores[i] >= 0.5]

    false_negatives.sort(key=lambda x: x['score'])
    false_positives.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nFound {len(false_negatives)} False Negatives (missed genuine pairs). Top 5 worst misses:")
    for item in false_negatives[:5]:
        print(f"  - Predicted Score: {item['score']:.4f} | Pair: {item['g1_id']} vs {item['g2_id']}")
        
    print(f"\nFound {len(false_positives)} False Positives (impostors accepted). Top 5 worst mistakes:")
    for item in false_positives[:5]:
        print(f"  - Predicted Score: {item['score']:.4f} | Pair: {item['g1_id']} vs {item['g2_id']}")

def run_xgboost_fingerprint_pipeline():
    print("="*60); print(" FINGERPRINT XGBOOST PIPELINE (FINAL TRAINING & SAVING) "); print("="*60)
    
    print("\n[1/5] Loading data..."); users = load_users_dictionary('processed_minutiae_data.pkl', True)
    analyzer = GraphMinutiae(users)
    print("\n[2/5] Building graphs..."); analyzer.graph_maker()
    print("\n[3/5] Creating pairs for evaluation..."); all_pairs = analyzer.create_graph_pairs()
    print("\n[4/5] Splitting users for disjoint sets...")
    train_users, val_users, test_users = analyzer.get_user_splits()
    train_pairs, _, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_users, val_users, test_users)
    
    print("\n[5/5] Preparing data and training XGBoost model...")
    X_train, y_train = prepare_data_for_xgboost(train_pairs)
    X_test, y_test = prepare_data_for_xgboost(test_pairs)

    print(f"\n--- Starting Hyperparameter Tuning on {X_train.shape[0]} training pairs ---")
    param_grid = {'max_depth': [5, 7], 'learning_rate': [0.05, 0.1], 'n_estimators': [300, 500],
                  'gamma': [0, 0.1], 'colsample_bytree': [0.7, 0.8]}
    xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_jobs=-1)
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=3)
    grid_search.fit(X_train, y_train)

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation AUC score: {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_

    # --- SAVE THE FINAL MODEL ---
    model_filename = 'fingerprint_verifier.joblib'
    print(f"\nSaving the best model to '{model_filename}'...")
    joblib.dump(best_model, model_filename)
    print("Model saved.")
    
    # Evaluate the final, tuned model, now with error analysis
    evaluate_xgboost_model(best_model, X_test, y_test, test_pairs)
    
    print("\n--- XGBOOST PIPELINE COMPLETE ---")

if __name__ == '__main__':
    run_xgboost_fingerprint_pipeline()