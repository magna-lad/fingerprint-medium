import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from load_save import *
from itertools import combinations, product
import random
    
class MinutiaeROCAnalyzer:
    def __init__(self, users_minutiae):
        """
        Args:
            users_minutiae: dict {user_id: [ [fingerprint1], [fingerprint2], ... ]}
                            Each fingerprint has a list of minutiaes
                            Each minutiae list is a list of minutiae [(x,y), type, angle].
        source: https://doi.org/10.1016/j.measen.2025.101809  -> for making feature vectors
        
        """

        self.users_minutiae= users_minutiae
        self.users_feature_vectors = {
            uid: {
                "fingers": {
                    hand: [
                        [None for _ in finger]   # len(impressions)
                        for finger in user_data["fingers"][hand]
                    ] for hand in ["L", "R"]
                }
            } for uid, user_data in users_minutiae.items()
        }
        #print(self.users_feature_vectors['000'])
        


    # find the neighbouring minutiae for a minutiae and make feature vectors to feed in the cnn algo
    
        '''
        convert all the minutiae from [(x,y), type, angle] to [x,y, type, angle]

        users = {
            "000": {
                "fingers": {
                    "L": [
                            {"finger":[numpy array],"minutiae":[minutiae array]},
                            {impression2},...],   # finger 0
                           [impr1, impr2, impr3, impr4, impr5],   # finger 1
                           [impr1, impr2, impr3, impr4, impr5],   # finger 2
                           [impr1, impr2, impr3, impr4, impr5] ], # finger 3
                    "R": [ [...], [...], [...], [...] ]
                }
            },
            ...
        }                                                                                                                  
        '''
        

    
    @staticmethod
    def extract_neighbors(minutiae, k=5):
        """
        Extract k nearest neighbors for each minutia point based on Euclidean distance of (x,y) coordinates.
    
        Parameters:
        minutiae: np.array of shape (N, 4) where columns represent [x, y, type, angle]
        k: int - number of nearest neighbors
    
        Returns:
        neighbors_indices: list of lists - each list contains indices of nearest neighbors for corresponding minutia
        neighbors_distances: list of lists - distances corresponding to neighbors_indices
        """
        #print(minutiae)
        #print(minutiae)
        coords = minutiae[:, :2]  # extract only (x,y) for neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords)  # +1 because the closest neighbor is itself
        distances, indices = nbrs.kneighbors(coords)
    
        # Remove the first neighbor (itself) for each minutia
        neighbors_indices = [list(idx[1:]) for idx in indices]
        neighbors_distances = [list(dist[1:]) for dist in distances]
    
        return neighbors_indices, neighbors_distances
    
    @staticmethod
    def compute_relative_angles(minutiae, neighbors_indices):
        """
        Compute relative angle differences between each minutia and its neighbors.
    
        Parameters:
        minutiae: np.array of shape (N, 4) - (x, y, type, angle)
        neighbors_indices: list of lists - neighbors for each minutia
    
        Returns:
        neighbors_relative_angles: list of lists - relative angle differences in degrees
        """
        neighbors_relative_angles = []
        for i, neighbors in enumerate(neighbors_indices):
            angle_ref = minutiae[i, 3]
            rel_angles = []
            for nbr_idx in neighbors:
                angle_nbr = minutiae[nbr_idx, 3]
                diff = abs(angle_ref - angle_nbr)
                diff = min(diff, 360 - diff)  # smallest angle difference
                rel_angles.append(diff)
            neighbors_relative_angles.append(rel_angles)
        return neighbors_relative_angles
    
    @staticmethod
    def create_feature_vectors(minutiae, neighbors_indices, neighbors_distances, neighbors_relative_angles):
        """
        Create feature vectors for each minutia combining own features and neighbors' distances and relative angle differences.
    
        Parameters:
        minutiae: np.array of shape (N, 4) - columns: [x, y, type, angle]
        neighbors_indices: list of lists
        neighbors_distances: list of lists
        neighbors_relative_angles: list of lists
    
        Returns:
        feature_vectors: np.array of shape (N, 4+2*k) - combined features per minutia
        """
        feature_vectors = []
        k = len(neighbors_indices[0]) if neighbors_indices else 0
    
        for i in range(len(minutiae)):
            own_features = minutiae[i].tolist()  # [x, y, type, angle]
            dist_features = neighbors_distances[i]
            angle_features = neighbors_relative_angles[i]
            combined = own_features + dist_features + angle_features
            feature_vectors.append(combined)
    
        return np.array(feature_vectors)
    
    # just to know the structure of self variables
    def print_structure(self,data, max_depth=3, indent=0):
        if max_depth == 0:
            print('  ' * indent + '...')
            return
        if isinstance(data, dict):
            print('  ' * indent + '{')
            for k, v in data.items():
                print('  ' * (indent + 1) + f'{repr(k)}:')
                self.print_structure(v, max_depth - 1, indent + 2)
            print('  ' * indent + '}')
        elif isinstance(data, list):
            print('  ' * indent + '[')
            n = min(3, len(data))
            for i in range(n):
                self.print_structure(data[i], max_depth - 1, indent + 1)
            if len(data) > n:
                print('  ' * (indent + 1) + '...')
            print('  ' * indent + ']')
        elif hasattr(data, 'shape') and hasattr(data, 'dtype'):
            print('  ' * indent + f'np.ndarray shape={data.shape} dtype={data.dtype}')
        else:
            print('  ' * indent + repr(data))


    def k_nearest_negihbors(self,k=5):
        '''
        input- minutiae list of a fingerprint

        output- feature vector containing [x,y,type,angle,neighbourhood distances, reference angles]
        '''
        #extract minutiae of a fingerprint
        #arrange them based on k nearest neighbours
        
        for user_id, user_data  in tqdm(self.users_minutiae.items(), desc="Processing users"):
                for hand,fingers in user_data["fingers"].items(): 
                    for finger_index, impressions in enumerate(fingers):
                        for impression_index, image in enumerate(impressions):

                            neighbors_indices, neighbors_distances = self.extract_neighbors(image["minutiae"], k=3)
                            neighbors_relative_angles = self.compute_relative_angles(image["minutiae"], neighbors_indices)
                            feature_vectors = self.create_feature_vectors(image["minutiae"], neighbors_indices, neighbors_distances, neighbors_relative_angles)
                            self.users_feature_vectors[user_id]["fingers"][hand][finger_index][impression_index] = feature_vectors
        #print(len(self.users_feature_vectors["000"]["fingers"]["L"][0][1]))



        
    
    def genuine_pairs_and_impostor_pairs(self,num_samples_per_genuine=1):
        '''
        input:
        {
          '000':
            {
              'fingers':
                {
                  'L':[
                        [feature_vectors_impression1],[feature_vectors_impression2]...],    # finger0
                        [feature_vectors_impression1],[feature_vectors_impression2]...],    # finger1
                        [feature_vectors_impression1],[feature_vectors_impression2]...],    # finger2
                        [feature_vectors_impression1],[feature_vectors_impression2]...], ]  # finger3
                    ...
                  'R':[same]
                    ...
                }
            }
          '001':
            .......
        }

        making pairs with impression images within the same finger list will be genuine pairs and all the other pairs will be  impostor pairs
        output:
        genuine and impostor pairs to train the model on

        '''
        genuine_pairs = []
        impostor_pairs = []
        subjects = list(self.users_feature_vectors.keys())

        # Generating genuine pairs (within the same finger impressions)
        for subject, vals in self.users_feature_vectors.items():
            fingers = vals.get('fingers', {})
            for hand, fingers_list in fingers.items():
                for finger_impressions in fingers_list:
                    for pair in combinations(finger_impressions, 2):
                        genuine_pairs.append((pair[0], pair[1], 1))

        # impostor fingerprint pairs
        subjects = list(self.users_feature_vectors.keys())
        #print(subjects)
        # Impostor pairs: match corresponding fingers/hands between different subjects
        for idx1, subject1 in enumerate(subjects):
            for subject2 in subjects[idx1+1:]:
                for hand in ['L', 'R']:
                    fingers1 = self.users_feature_vectors[subject1]['fingers'].get(hand, [])
                    fingers2 = self.users_feature_vectors[subject2]['fingers'].get(hand, [])
                    for finger_idx in range(min(len(fingers1), len(fingers2))):
                        impressions1 = fingers1[finger_idx]
                        impressions2 = fingers2[finger_idx]
                        # All combinations between this finger (cross-subject, impostor)
                        for imp1 in impressions1:
                            for imp2 in impressions2:
                                impostor_pairs.append((imp1, imp2, 0))

        # Downsample impostor pairs for balance, if needed
        if len(impostor_pairs) > num_samples_per_genuine * len(genuine_pairs):
            impostor_pairs = random.sample(
                impostor_pairs, num_samples_per_genuine * len(genuine_pairs))

        # Combine and save
        labeled_pairs = []
        print(len(genuine_pairs), len(impostor_pairs))
        labeled_pairs = genuine_pairs + impostor_pairs
        save_users_dictionary(labeled_pairs,"labeled_pairs.pkl")
        #print((labeled_pairs[0][0][2]))

        return labeled_pairs # to train the model on

    





users=load_users_dictionary('processed_minutiae_data.pkl',True)

#print(users)

ko=MinutiaeROCAnalyzer(users)
ko.k_nearest_negihbors(k=3)
print('saatata')
ko.genuine_pairs_and_impostor_pairs()

