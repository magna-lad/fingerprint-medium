import matplotlib.pyplot as plt
from minutia_loader import minutiaLoader
from skeleton_maker import skeleton_maker
from reader import load_users
from tqdm import tqdm
from minutiaeExtractor import minutiaeExtractor
from roc_scores import perform_roc_analysis
import pickle
import os
from datetime import datetime

def load_users_dictionary(filename="processed_skeletons.pkl"):
    """Load the saved users dictionary"""
    
    cache_dir = "biometric_cache"
    filepath = os.path.join(cache_dir, filename)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
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
    """Save the processed users dictionary"""
    
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
    
    return filepath

def main():
    data_dir = r"C:\Users\kound\OneDrive\Desktop\10Classes"
    
    # Step 1: Load or process skeleton data
    print("Checking for cached skeleton data...")
    users = load_users_dictionary("processed_skeletons.pkl")
    
    if users is None:
        print("No cached skeleton data found. Processing from scratch...")
        
        # Load original data
        users = load_users(data_dir)
        
        # Process skeletons - Fixed loop structure
        for user_id, finger_dic in tqdm(users.items(), desc="Processing users"):
            for idx, image in enumerate(finger_dic['finger']):  # Fixed: direct access
                try:
                    fingerprint = minutiaLoader(image)
                    skeleton_image = skeleton_maker(
                        fingerprint.normalised_img,
                        fingerprint.segmented_img,
                        fingerprint.norm_img,
                        fingerprint.mask,
                        fingerprint.block
                    )
                    
                    # Process skeleton
                    skeleton_image.fingerprintPipeline()
                    
                    # Replace original with processed skeleton
                    users[user_id]['finger'][idx] = skeleton_image.skeleton
                    
                except Exception as e:
                    print(f"Error processing user {user_id}: {e}")
        
        # Save processed skeleton data
        save_users_dictionary(users, "processed_skeletons.pkl")
        
    else:
        print("Using cached skeleton data - skipping skeleton generation.")
    
    # Step 2: Load or extract minutiae data
    print("\nChecking for cached minutiae data...")
    users_with_minutiae = load_users_dictionary("processed_with_minutiae.pkl")
    
    if users_with_minutiae is None:
        print("No cached minutiae data found. Extracting minutiae from skeletons...")
        
        # Extract minutiae from skeletons
        for user_id, finger_dic in tqdm(users.items(), desc="Extracting minutiae"):
            minutiae_list = []  # Store minutiae for this user
            
            for idx, skeleton_image in enumerate(finger_dic['finger']):
                try:
                    minutiae_extractor = minutiaeExtractor(skeleton_image)
                    minutiae_points = minutiae_extractor.extract()
                    minutiae_list.append(minutiae_points)
                    
                except Exception as e:
                    print(f"Error extracting minutiae for user {user_id}, image {idx}: {e}")
                    minutiae_list.append([])  # Empty list for failed extractions
            
            # Store minutiae data alongside skeleton data
            users[user_id]['minutiae'] = minutiae_list
        
        # Save the complete data (skeletons + minutiae)
        save_users_dictionary(users, "processed_with_minutiae.pkl")
        
        # Use the newly processed data
        final_users_data = users
        
    else:
        print("Using cached minutiae data - skipping minutiae extraction.")
        # Use the cached data that already contains minutiae
        final_users_data = users_with_minutiae
    
    # Step 3: Display processing summary
    print("\n=== Processing Summary ===")
    total_images = 0
    total_minutiae = 0
    
    for user_id, user_data in final_users_data.items():
        user_images = len(user_data['finger'])
        user_minutiae = sum(len(minutiae) for minutiae in user_data.get('minutiae', []))
        total_images += user_images
        total_minutiae += user_minutiae
        print(f"User {user_id}: {user_images} images, {user_minutiae} total minutiae")
    
    print(f"\nOverall: {total_images} images processed, {total_minutiae} total minutiae extracted")
    
    # Step 4: Perform ROC analysis
    print("\n" + "="*50)
    print("PERFORMING ROC ANALYSIS")
    print("="*50)
    
    roc_analyzer = perform_roc_analysis(final_users_data)  # Use correct dataset
    
    # Save ROC results
    metrics = roc_analyzer.get_performance_metrics()
    
    # Save detailed results
    results_summary = {
        'users_processed': len(final_users_data),
        'performance_metrics': metrics,
        'genuine_scores': roc_analyzer.genuine_scores,
        'impostor_scores': roc_analyzer.impostor_scores,
        'roc_data': roc_analyzer.roc_data
    }
    
    # Ensure directory exists and save results
    os.makedirs('biometric_cache', exist_ok=True)
    with open('biometric_cache/roc_analysis_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"\nROC analysis complete! Results saved to biometric_cache/")
    
    return final_users_data, roc_analyzer

if __name__ == '__main__':
    main()
