import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import os
import pickle
#from minutiae_repair import minutiae_repair
class MinutiaeROCAnalyzer:
    def __init__(self, users_minutiae):
        """
        Args:
            users_minutiae: dict {user_id: [ [fingerprint1], [fingerprint2], ... ]}
                            Each fingerprint has a list of minutiaes
                            Each minutiae list is a list of minutiae [(x,y), type, angle].
        https://doi.org/10.1016/j.measen.2025.101809 
        
        """

        self.users_minutiae= users_minutiae

    # find the neighbouring minutiae for a minutiae and make feature vectors to feed in the cnn algo
    # earlier mistake
    # made the structure as [(x,y), type, angle] but needs to be [x,y, type, angle] for error-less numpy conversion

    def minutiae_repair(self):
        '''
        convert all the minutiae from [(x,y), type, angle] to [x,y, type, angle]

        user list
        {
        000: {'minutiae':[    **list of all fingerprints minutiaes
                            [ **minutiae of a fingerprint],
                            [ **minutiae of a fingerprint],
                            .......
                           ]
               },
        001 :{'minutiae':[    **list of all fingerprints minutiaes
                            [ **minutiae of a fingerprint],
                            [ **minutiae of a fingerprint],
                            .......
                           ]
               },
        ......
        }
        '''
        for user_id,all_data in tqdm(self.users_minutiae.items()):
            for minutiae_head,all_fingers_per_user in all_data.items():
               for finger_idx, finger in enumerate(all_fingers_per_user):
                    new_finger = []
                    for minutia in finger:
                        (x, y), typ, angle = minutia
                        arr = np.array([x, y, typ, angle])
                        new_finger.append(arr)
                    self.users_minutiae[user_id][minutiae_head][finger_idx] = new_finger

        print(self.users_minutiae)

    def k_nearest_negihbors(self,k=5):
        '''
        within a fingerprint find the nearest minutiae points with euclidean distance
        '''
        #extract minutiae of a fingerprint
        #arrange them based on k nearest neighbours
        #print((self.users_minutiae.values()))
        
        







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
        #total_skeletons = sum(len(finger_data['finger']) for finger_data in users.values())
        #print(f"Total skeletons: {total_skeletons}")
        
        return users
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
users=load_users_dictionary('processedSliced_minutiae_data.pkl')

#print(users)

ko=MinutiaeROCAnalyzer(users)
ko.minutiae_repair()