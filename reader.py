import os
import cv2


# initiate loading the users
# returns path to the users

def load_users(data_dir):
    users = {}
    all_images=[]
    for uid in os.listdir(data_dir):
        uid_path = os.path.join(data_dir, uid) # entering the numbered folders
        if not os.path.isdir(uid_path): 
            continue
        
        imgs = []
        user_img=[]

        for hand in ['L', 'R']: # folder structure
            hand_dir = os.path.join(uid_path, hand) 
            if os.path.isdir(hand_dir):
                for f in os.listdir(hand_dir): # entering the hand folder
                    if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                        img_path = os.path.join(hand_dir, f)
                        img = cv2.imread(img_path, 0) # read in grayscale
                        if img is not None:
                            user_img.append(img)
                            #img = cv2.resize(img)
                            all_images.append(img)

        users[uid] = {'finger': user_img} #list of users with fingers array representation
    
    return users