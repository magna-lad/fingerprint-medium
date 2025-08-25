#import pickle
#
## Path to your pickle file
##file_path = r"./processed_minutiae_data.pkl"
#
#file_path = r"C:\Users\kound\OneDrive\Desktop\fingerprint_mine\biometric_cache\processed_skeletons.pkl"
## Load the contents
#with open(file_path, 'rb') as f:
#    data = pickle.load(f)
#
## Now you can inspect the structure
#print(type(data))               # e.g. dict
#print(len(data))                # Number of users
#
## Example: Print first user ID and its data
#for user_id, user_data in data.items():
#    print("User ID:", user_id)
#    print("Keys:", user_data.keys())  # Should include 'finger', maybe 'minutiae'
#    
#    print(f"Number of fingerprint images: {len(user_data['finger'])}")
#    
#    # If minutiae exists
#    if 'minutiae' in user_data:
#        print(f"Number of minutiae lists: {(user_data['minutiae'][0])}")
#    
#    break  # Remove this if you want to loop over all users
#

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm

# Load the data
file_path = r"C:\Users\kound\OneDrive\Desktop\fingerprint_mine\biometric_cache\processed_minutiae_data.pkl"

with open(file_path, 'rb') as f:
    users = pickle.load(f)
    #print(data)

# Draw and show for one user
for user_id, user_data  in tqdm.tqdm(users.items(), desc="Processing users"):
    for hand,fingers in user_data["fingers"].items(): 
        for finger_index, impressions in enumerate(fingers):
            for impression_index, image in enumerate(impressions):
                print("start")
                print("User ID:", user_id)

                # Get the first skeleton and its minutiae
                skeleton = users[user_id]["fingers"][hand][finger_index][impression_index]['finger']
                minutiae = users[user_id]["fingers"][hand][finger_index][impression_index]['minutiae']  # list of (x, y)
                print(minutiae)
                # Convert skeleton to BGR for color drawing
                if len(skeleton.shape) == 2:
                    skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
                else:
                    skeleton_bgr = skeleton.copy()

                # Draw red circles for minutiae
                for pt in minutiae:
                    x, y = pt[0],pt[1]
                    cv2.circle(skeleton_bgr, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)  # red dot
                # Display using matplotlib
                plt.figure(figsize=(6, 6))
                plt.imshow(cv2.cvtColor(skeleton_bgr, cv2.COLOR_BGR2RGB))
                plt.title(f"User {user_id} - Minutiae on Skeleton")
                plt.axis('off')
                plt.show()
    
      # Remove to process all users




#import pickle
#import numpy as np
#import matplotlib.pyplot as plt
#import math
#
## Function to plot minutiae and orientation on skeleton
#def plot_minutiae_orientations(skeleton, minutiae):
#    # Convert grayscale skeleton to RGB for color drawing
#    if len(skeleton.shape) == 2:
#        skeleton_rgb = np.stack([skeleton]*3, axis=-1)
#    else:
#        skeleton_rgb = skeleton.copy()
#
#    fig, ax = plt.subplots(figsize=(8, 8))
#    ax.imshow(skeleton_rgb, cmap='gray')
#
#    for m in minutiae:
#        (x, y), angle, typ = m  # unpack
#        ax.plot(x, y, 'ro')  # red dot for minutia
#
#        # Arrow for orientation (assume angle in radians)
#        length = 15
#        end_x = x + length * np.cos(angle*math)
#        end_y = y + length * np.sin(angle*math)
#        ax.arrow(x, y, end_x - x, end_y - y, head_width=4, head_length=6, fc='lime', ec='lime')
#
#    ax.set_title('Minutiae and Orientation Arrows on Skeleton')
#    ax.axis('off')
#    plt.show()
#
## Load the data (update with your actual file path)
#file_path = r"C:\Users\kound\OneDrive\Desktop\fingerprint_mine\biometric_cache\processed_minutiae_data.pkl"
#
#with open(file_path, 'rb') as f:
#    data = pickle.load(f)
#
## Plot for the first user and second fingerprint sample
#for user_id, user_data in data.items():
#    print("User ID:", user_id)
#    
#    skeleton = user_data['finger'][1]  # Assuming this is the skeleton image
#    minutiae = user_data['minutiae'][1]  # Assuming list of [((x,y), angle, type)]
#    
#    plot_minutiae_orientations(skeleton, minutiae)
#    #break  # Remove to process all users
