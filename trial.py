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

# Load the data
file_path = r"C:\Users\kound\OneDrive\Desktop\fingerprint_mine\processed_minutiae_data.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Draw and show for one user
for user_id, user_data in data.items():
    print("User ID:", user_id)

    # Get the first skeleton and its minutiae
    skeleton = user_data['finger'][1]
    minutiae = user_data['minutiae'][1]  # list of (x, y)
    
    # Convert skeleton to BGR for color drawing
    if len(skeleton.shape) == 2:
        skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    else:
        skeleton_bgr = skeleton.copy()

    # Draw red circles for minutiae
    for pt in minutiae:
        x, y = pt[0]
        cv2.circle(skeleton_bgr, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)  # red dot

    # Display using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(skeleton_bgr, cv2.COLOR_BGR2RGB))
    plt.title(f"User {user_id} - Minutiae on Skeleton")
    plt.axis('off')
    plt.show()
    
      # Remove to process all users
