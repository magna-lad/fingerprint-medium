import os
import cv2
from collections import defaultdict

# initiate loading the users
# returns path to the users

def load_users(data_dir):
    '''
    expected input-
    a folder containing several users with L and R folders containing images of fingers
    with 5 impressions per finger
    
    output-
    a dictionary of users with key- user id
    item- for each user, a dictionary, will contain finger key with 
                                                        item- a list with two nested lists for left and right
                                                        each l and f folder contains nested list with each list containing
                                                        minutiaes of impressions of the same finger
        same structrure of masks of the image for preprocessing
    '''
    users = {}
    for uid in os.listdir(data_dir):
        uid_path = os.path.join(data_dir, uid) # entering the numbered folders
        if not os.path.isdir(uid_path):  # error handling
            continue
        
        

        for hand in ['L', 'R']: # folder structure
            img=[]
            count=0
            hand_dir = os.path.join(uid_path, hand) 
            if not os.path.isdir(hand_dir):
                continue
            if os.path.isdir(hand_dir):
                for f in os.listdir(hand_dir): # entering the L,R hand folder
                    img_buffer=[]
                    if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')) :
                        continue    

                    try :
                        finger_idx=int(f(5))
                    except (IndexError, ValueError):
                        continue  # skip files that don't match pattern
                    img_path = os.path.join(hand_dir, f)
                    img = cv2.imread(img_path, 0)

                    if img is not None:
                        fingers[hand][finger_idx].append(img)
                        #print(f[5])
        if any(fingers[hand] for hand in ['L', 'R']):
            # Optionally convert defaultdict to dict for JSON-serializability
            users[uid] = {'fingers': {h: dict(fingers[h]) for h in fingers}}
    print(users)
    return users

load_users(r"C:\Users\kound\OneDrive\Desktop\10Classes\5classes")

