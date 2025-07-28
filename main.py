import matplotlib.pyplot as plt
from minutia_loader import minutiaLoader
from skeleton_maker import skeleton_maker
from reader import load_users
from tqdm import tqdm
#from minutiaeExtractor import minutiaeExtractor
from roc_scores import perform_roc_analysis
import pickle
import os
from datetime import datetime
from minutiae_filter import minutiae_filter

def load_users_dictionary(filename):
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

def save_users_dictionary(users, filename):
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
    data_dir = r"C:\Users\kound\OneDrive\Desktop\10Classes\5classes"
    
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
                        fingerprint.block,
                    )
                    
                    # Process skeleton
                    skeleton_image.fingerprintPipeline()
                    
                    # Replace original with processed skeleton
                    users[user_id]['finger'][idx] = skeleton_image.skeleton
                    # Initialize 'minutiae' list if not already present
                    if 'minutiae' not in users[user_id]:
                        users[user_id]['minutiae'] = [None] * len(users[user_id]['finger'])

                    # Store extracted minutiae
                    users[user_id]['minutiae'][idx] = skeleton_image.minutiae_list

                    if 'mask' not in users[user_id]:
                        users[user_id]['mask'] = [None] * len(users[user_id]['finger'])
                    
                    users[user_id]['mask'][idx] = fingerprint.mask
                    
                except Exception as e:
                    print(f"Error processing user {user_id}: {e}")
        # Save processed skeleton data
        save_users_dictionary(users, "processed_skeletons.pkl")
        
    else:
        print("Using cached skeleton data - skipping skeleton generation.")
    




    
    #filter the minutiaes    
    processed_users = load_users_dictionary("processed_minutiae_data.pkl")

    if processed_users is None:
        print("No processed users data found. Processing from scratch...")
        
        processed_users = {}

        try:
            for user_id, image_dic in tqdm(users.items(), desc="Filtering users"):
                fingerprint_list = image_dic['finger']     # List of 40 skeletons
                minutiae_list = image_dic['minutiae']      # List of 40 minutiae sets
                mask_list = image_dic['mask']

                mf = minutiae_filter(fingerprint_list, minutiae_list,mask_list)
                filtered_fingers, filtered_minutiae = mf.filter_all() # add parameters min_distance=5, border_margin=10

                processed_users[user_id] = {
                    'finger': filtered_fingers,
                    'minutiae': filtered_minutiae
                }

            #print(processed_users)
            # save to disk
            save_users_dictionary(processed_users, "processed_minutiae_data.pkl")
        except Exception as e:
            print(f"Error processing user : {e}")

    else:
        print("Using processed minutiae data - skipping minutiae filtering.")
        


    # Step 3: Display processing summary
    users_filtered = load_users_dictionary("processed_minutiae_data.pkl")
    #print(users_filtered)
    print("\n=== Processing Summary ===")
    total_images = 0
    total_minutiae = 0
    
    for user_id, user_data in users_filtered.items():
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
    
    roc_analyzer = perform_roc_analysis(users_filtered)  # Use correct dataset
    
    # Save ROC results
    metrics = roc_analyzer.get_performance_metrics()
    
    # Save detailed results
    results_summary = {
        'users_processed': len(users_filtered),
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
    
    return processed_users, roc_analyzer

if __name__ == '__main__':
    main()
