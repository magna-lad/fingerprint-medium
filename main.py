import matplotlib.pyplot as plt
from minutia_loader import minutiaLoader
from skeleton_maker import skeleton_maker
from reader import load_users
#from simple_roc_analysis import ImprovedSkeletonROCAnalyzer  # Import simple function
from tqdm import tqdm
from scores_roc import scores_roc



import pickle
import os
from datetime import datetime


def load_users_dictionary(filename="processed_skeletons.pkl"):
    """Load the saved users dictionary"""
    
    cache_dir = "biometric_cache"
    filepath = os.path.join(cache_dir, filename)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        # Load with pickle
        with open(filepath, 'rb') as f:
            users = pickle.load(f)
        
        print(f"Users dictionary loaded successfully!")
        print(f"Location: {filepath}")
        print(f"Users loaded: {len(users)}")
        
        # Verify structure and show summary
        total_skeletons = sum(len(finger_data['finger']) for finger_data in users.values())
        print(f"Total skeletons: {total_skeletons}")
        
        return users
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def save_users_dictionary(users, filename="processed_skeletons.pkl"):
    """Save the processed users dictionary with skeletons"""
    
    # Create directory if it doesn't exist
    os.makedirs("biometric_cache", exist_ok=True)
    
    # Full filepath
    filepath = os.path.join("biometric_cache", filename)
    
    # Save with pickle
    with open(filepath, 'wb') as f:
        pickle.dump(users, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Users dictionary saved successfully!")
    print(f"Location: {filepath}")
    print(f"Users saved: {len(users)}")
    
    # Print summary
    total_skeletons = sum(len(finger_data['finger']) for finger_data in users.values())
    print(f"Total skeletons: {total_skeletons}")
    
    return filepath

def main():
    data_dir = r"C:\Users\kound\OneDrive\Desktop\10Classes"
    
    # Try to load existing processed data first
    print("Checking for cached processed data")
    users = load_users_dictionary("processed_skeletons.pkl")
    
    if users is None:
        print("No cached data found. Processing from scratch...")
        
        # Load original data
        users = load_users(data_dir)

        
        # Your existing processing loop
        for user_id, finger_dic in tqdm(users.items(), desc="Processing users"):
            for finger, finger_list in finger_dic.items():
                for idx, image in enumerate(finger_list):
                    try:
                        fingerprint = minutiaLoader(image)
                        skeleton_image = skeleton_maker(
                            fingerprint.normalised_img,
                            fingerprint.segmented_img,
                            fingerprint.norm_img,
                            fingerprint.mask,
                            fingerprint.block
                        )
                        
                        # Your skeleton processing
                        skeleton_image.fingerprintPipeline()
                        
                        # Replace original with processed skeleton
                        users[user_id]['finger'][idx] = skeleton_image.skeleton
                        
                    except Exception as e:
                        print(f"Error processing user {user_id}: {e}")
        
        # Save processed data automatically
        save_users_dictionary(users, "processed_skeletons.pkl")
        
    else:
        print("Using cached processed data - skipping skeleton generation.")
    
    # Continue with ROC analysis using saved data
    enhanced_analyzer = scores_roc(users)
    results = enhanced_analyzer.get_summary_report()


if __name__ == '__main__':
    main()