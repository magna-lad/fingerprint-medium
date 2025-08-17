import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
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
        self.users_feature_vector = {uid: {head: [None] * len(fingers) for head, fingers in data.items()}
                                     for uid, data in users_minutiae.items()}


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
        for user_id,all_data in tqdm(self.users_minutiae.items(),desc="Repairing minutiae"):
            for minutiae_head,all_fingers_per_user in all_data.items():
               for finger_idx, finger in enumerate(all_fingers_per_user):  
                    new_finger = []
                    for minutia in finger:
                        (x, y), typ, angle = minutia
                        new_finger.append([x, y, typ, angle])
                    self.users_minutiae[user_id][minutiae_head][finger_idx] = np.array(new_finger)

        print(self.users_minutiae)

    
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
    

    def k_nearest_negihbors(self,k=5):
        '''
        input- minutiae list of a fingerprint

        output- feature vector containing [x,y,type,angle,neighbourhood distances, reference angles]
        '''
        #extract minutiae of a fingerprint
        #arrange them based on k nearest neighbours
        #print((self.users_minutiae.values()))
        #print(self.users_minutiae)
        for user_id,all_data in tqdm(self.users_minutiae.items(), desc='feature vector generation'):
            for minutiae_head,all_fingers_per_user in all_data.items():
               for finger_idx, finger in enumerate(all_fingers_per_user):
                    
                    neighbors_indices, neighbors_distances = self.extract_neighbors(finger, k=3)
                    neighbors_relative_angles = self.compute_relative_angles(finger, neighbors_indices)
                    feature_vectors = self.create_feature_vectors(finger, neighbors_indices, neighbors_distances, neighbors_relative_angles)
                    self.users_feature_vector[user_id][minutiae_head][finger_idx] = feature_vectors

        print(self.users_feature_vector)





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
ko.k_nearest_negihbors(k=3)