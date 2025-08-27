import matplotlib.pyplot as plt
from minutia_loader import minutiaLoader
from skeleton_maker import skeleton_maker
from reader import load_users
from tqdm import tqdm
#from minutiaeExtractor import minutiaeExtractor
#from roc_scores import MinutiaeROCAnalyzer
import pickle
import os
from datetime import datetime
from minutiae_filter import minutiae_filter
from load_save import *

def main():
    data_dir = r"C:\Users\kound\OneDrive\Desktop\10Classes\5classes"
    
    # Step 1: Load or process skeleton data
    print("Checking for cached skeleton data...")
    users = load_users_dictionary("processed_skeletons.pkl")
    
    '''
    structure-
    users = {
            "000": {
                "fingers": {
                    "L": [ [impr1, impr2, impr3, impr4, impr5],   # finger 0
                           [impr1, impr2, impr3, impr4, impr5],   # finger 1
                           [impr1, impr2, impr3, impr4, impr5],   # finger 2
                           [impr1, impr2, impr3, impr4, impr5] ], # finger 3
                    "R": [ [...], [...], [...], [...] ]
                }
            },
            ...
        }
    '''
    
    if users is None:
        print("No cached skeleton data found. Processing from scratch...")
        
        # Load original data
        users = load_users(data_dir)
        #print(users)
        # Process skeletons - Fixed loop structure
        for user_id, user_data  in tqdm(users.items(), desc="Processing users"):
            for hand,fingers in user_data["fingers"].items(): 
                for finger_index, impressions in enumerate(fingers):
                    for impression_index, image in enumerate(impressions):

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


                            users[user_id]["fingers"][hand][finger_index][impression_index] = {
                                    "skeleton": skeleton_image.skeleton,
                                    "minutiae": skeleton_image.minutiae_list,
                                    "mask": fingerprint.mask
                                }

                        except Exception as e:
                            print(f"Error processing {user_id}-{hand}-f{finger_index}-i{impression_index}: {e}")
        # Save processed skeleton data
        #print(users)
        save_users_dictionary(users, "processed_skeletons.pkl")
        
    else:
        print("Using cached skeleton data - skipping skeleton generation.")
    
    #print(users.items()['fingers']['L'])




    # improve this


    
    #filter the minutiaes    
    processed_users = load_users_dictionary("processed_minutiae_data.pkl")

    '''
    structure-
    users = {
            "000": {
                "fingers": {
                    "L": [ [{
                            "skeleton"=[],
                            "minutiae"=[],
                            "mask"=[]
                            },
                            {
                            "skeleton"=[],
                            "minutiae"=[],
                            "mask"=[]
                            }, {
                            "skeleton"=[],
                            "minutiae"=[],
                            "mask"=[]
                            }, {
                            "skeleton"=[],
                            "minutiae"=[],
                            "mask"=[]
                            }, {
                            "skeleton"=[],
                            "minutiae"=[],
                            "mask"=[]
                            }],   # finger 0
                           [impr1, impr2, impr3, impr4, impr5],   # finger 1
                           [impr1, impr2, impr3, impr4, impr5],   # finger 2
                           [impr1, impr2, impr3, impr4, impr5] ], # finger 3
                    "R": [ [...], [...], [...], [...] ]
                }
            },
            ...
        }
    '''
    if processed_users is None:
        print("No processed users data found. Processing from scratch...")
        
        #processed_users = {}

        try:


            for user_id, user_data  in tqdm(users.items(), desc="Processing users"):
                for hand,fingers in user_data["fingers"].items(): 
                    for finger_index, impressions in enumerate(fingers):
                        for impression_index, image in enumerate(impressions):
                            skeleton_list = image['skeleton']
                            minutiae_list = image['minutiae']
                            mask_list = image['mask']
                            mf = minutiae_filter(skeleton_list, minutiae_list,mask_list)
                            filtered_fingers, filtered_minutiae = mf.filter_all() # add parameters min_distance=5, border_margin=10

                            users[user_id]["fingers"][hand][finger_index][impression_index] = {
                            'finger': filtered_fingers,
                            'minutiae': filtered_minutiae
                            } 

            #print(processed_users)
            # save to disk
            #print(processed_users)
            save_users_dictionary(users, "processed_minutiae_data.pkl")
        except Exception as e:
            print(f"Error processing user : {e}")

    else:
        print("Using processed minutiae data - skipping minutiae filtering.")
        

    #print(users.keys())
    #Step 3: Display processing summary
    users_filtered = load_users_dictionary("processed_minutiae_data.pkl")
    #print(users_filtered)
    print("\n=== Processing Summary ===")
    total_images = 0
    total_minutiae = 0
    
    for user_id, user_data  in tqdm(users.items(), desc="Processing users"):
        for hand,fingers in user_data["fingers"].items(): 
            for finger_index, impressions in enumerate(fingers):
                user_images = len(users[user_id]["fingers"][hand][finger_index])
                total_images += user_images
                user_minutiae=0
                for impression_index, image in enumerate(impressions):
                    num_minutiae = len(image["minutiae"])
                    user_minutiae+=num_minutiae
                    #print(user_minutiae)
                    #total_minutiae += user_minutiae
                    print(f"User {user_id}: {user_images} images, {user_minutiae} total minutiae")
    

    #print(users_filtered.values())
    print(f"\nOverall: {total_images} images processed, {total_minutiae} total minutiae extracted")
    
    # Step 4: Perform ROC analysis
    print("\n" + "="*50)
    print("PERFORMING ROC ANALYSIS")
    print("="*50)
#    
    

    ## now pass a new dictionary with user id as the key and minutiae as the items
    ## slice the users_filtered dictionary 
    #users_filtered_sliced = {}
    #for user_id, value_dic in users_filtered.items():
    #    users_filtered_sliced[user_id] = {'minutiae': value_dic['minutiae']}
#
    #save_users_dictionary(users_filtered_sliced, "processedSliced_minutiae_data.pkl")
#
    #for user_id, user_data in users_filtered_sliced.items():
    #    print("User ID:", user_id)
#
    #    # Get the first skeleton and its minutiae
    #    #skeleton = user_data['finger'][1]
    #    minutiae = user_data['minutiae']  # list of (x, y)
    #    print(len(minutiae))
#
    #print(users_filtered_sliced)
    #analyzer = MinutiaeROCAnalyzer(users_filtered_sliced,
    #                               distance_threshold=5,
    #                               angle_threshold=10,
    #                               match_threshold=3)
    #analyzer.compute_all_scores()
    #analyzer.generate_roc_curve()
    #analyzer.print_performance()
    #analyzer.plot_roc_curve()

 
    #results_summary = {
    #    'users_processed': len(users_filtered),
    #    'performance_metrics': metrics,
    #    'genuine_scores': roc_analyzer.genuine_scores,
    #    'impostor_scores': roc_analyzer.impostor_scores,
    #    'roc_data': roc_analyzer.roc_data
    #}
    
    # Ensure directory exists and save results
    #os.makedirs('biometric_cache', exist_ok=True)
    #with open('biometric_cache/roc_analysis_results.pkl', 'wb') as f:
    #    pickle.dump(results_summary, f)
    #


    #*****
    #print(f"\nROC analysis complete! Results saved to biometric_cache/")
    #
    #return processed_users#, roc_analyzer

if __name__ == '__main__':
    main()
