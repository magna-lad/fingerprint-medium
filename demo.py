import joblib
import numpy as np

# --- Local Imports ---
from graph_minutiae import GraphMinutiae
from xgboost_feature_extractor import create_feature_vector_for_pair
from load_save import load_users_dictionary

class FingerprintVerifier:
    def __init__(self, model_path='fingerprint_verifier.joblib'):
        """
        Initializes the verifier by loading the trained XGBoost model.
        """
        print(f"Loading model from {model_path}...")
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"ERROR: Model file not found at '{model_path}'.")
            print("Please run the 'run_xgboost_pipeline.py' script first to train and save the model.")
            exit()
            
        # We need an instance of GraphMinutiae to use its _build_single_graph method
        self.graph_builder = GraphMinutiae(users_minutiae={})
        print("Model loaded successfully.")

    def verify(self, minutiae1, minutiae2):
        """
        Takes two sets of minutiae, verifies them, and returns a similarity score.
        A score closer to 1.0 means a more likely match.
        """
        # 1. Convert raw minutiae arrays into graph objects
        graph1 = self.graph_builder._build_single_graph(minutiae1, "probe_1")
        graph2 = self.graph_builder._build_single_graph(minutiae2, "candidate_1")

        if graph1 is None or graph2 is None:
            print("Warning: Could not build a valid graph from one of the fingerprints (likely too few minutiae).")
            return 0.0

        # 2. Use the feature extractor to create the vector representing the pair's difference
        feature_vector = create_feature_vector_for_pair(graph1, graph2)
        
        # 3. Use the loaded model to predict the probability of the pair being a match
        # The model expects a 2D array, so we reshape our single feature vector
        # [0, 1] selects the probability for the "positive" class (label 1 = genuine)
        score = self.model.predict_proba(feature_vector.reshape(1, -1))[0, 1]
        
        return score

if __name__ == '__main__':
    # --- Example showing how to use the FingerprintVerifier ---
    
    # 1. Initialize the verifier. This loads the model you trained and saved.
    verifier = FingerprintVerifier()

    # 2. Load some example data to test the verifier
    print("\nLoading sample fingerprint data for demo...")
    try:
        users = load_users_dictionary('processed_minutiae_data.pkl', True)
    except FileNotFoundError:
        print("ERROR: Could not find 'processed_minutiae_data.pkl'.")
        print("Please ensure the dataset is available in the 'biometric_cache' directory.")
        exit()
    
    # 3. Select minutiae for a genuine pair (same person, same finger, different impressions)
    minutiae_genuine_1 = users[100]['fingers']['Left'][0][0]['minutiae']
    minutiae_genuine_2 = users[100]['fingers']['Left'][0][1]['minutiae']
    
    # 4. Select minutiae for an impostor pair (different people)
    minutiae_impostor_1 = users[101]['fingers']['Right'][2][0]['minutiae']
    minutiae_impostor_2 = users[102]['fingers']['Right'][3][2]['minutiae']
    
    # 5. Get the similarity scores from the verifier
    genuine_score = verifier.verify(minutiae_genuine_1, minutiae_genuine_2)
    impostor_score = verifier.verify(minutiae_impostor_1, minutiae_impostor_2)
    
    print("\n" + "="*40)
    print("---            VERIFICATION RESULTS            ---")
    print("="*40)
    print(f"Genuine Pair Similarity Score: {genuine_score:.4f}")
    print(f"Impostor Pair Similarity Score: {impostor_score:.4f}")

    # 6. Make a decision based on a threshold (e.g., 0.5 or the EER threshold)
    threshold = 0.5 
    print(f"\nDecision with threshold = {threshold}:")
    print(f"  -> Is the genuine pair a match?  {'YES' if genuine_score >= threshold else 'NO'}")
    print(f"  -> Is the impostor pair a match? {'YES' if impostor_score >= threshold else 'NO'}")
    print("="*40)