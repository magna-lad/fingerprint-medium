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
import torch
from torch_geometric.data import Data
    
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

    # building graph from the knn maps
    def build_graph_from_minutiae(self,feature_vectors, neighbors_indices, neighbors_distances, neighbors_relative_angles):
        edges, edge_features = [], []
        for i, neighbors in enumerate(neighbors_indices):
            for nbr, dist, rel_ang in zip(neighbors, neighbors_distances[i], neighbors_relative_angles[i]):
                edges.append([i, nbr])
                edge_features.append([dist, rel_ang])

        # Convert lists into PyTorch tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        x = torch.tensor(feature_vectors, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    

    def graph_maker(self):
        """Build graph representations for all fingerprint impressions."""
        fingerprint_graphs = []
        graph_metadata = {}  # For easy lookup

        print("Building graphs from minutiae...")

        for uid, udata in tqdm(self.users_feature_vectors.items(), desc="Processing users"):
            for hand, fingers in udata['fingers'].items():
                for finger_idx, finger in enumerate(fingers):
                    for impression_idx, impression in enumerate(finger):
                        if impression is not None and len(impression) > 0:
                            minutiae = impression[:, :4]

                            # Build KNN structure
                            neighbors_indices, neighbors_distances = self.extract_neighbors(minutiae, k=3)
                            neighbors_relative_angles = self.compute_relative_angles(minutiae, neighbors_indices)

                            # Create graph
                            graph = self.build_graph_from_minutiae(
                                minutiae,
                                neighbors_indices,
                                neighbors_distances,
                                neighbors_relative_angles
                            )

                            # Create unique identifier for this impression
                            graph_id = f"{uid}_{hand}_{finger_idx}_{impression_idx}"

                            # Store with metadata
                            graph_info = {
                                'graph': graph,
                                'user_id': uid,
                                'hand': hand,
                                'finger_idx': finger_idx,
                                'impression_idx': impression_idx,
                                'graph_id': graph_id
                            }

                            fingerprint_graphs.append(graph_info)
                            graph_metadata[graph_id] = graph_info

        print(f"Built {len(fingerprint_graphs)} graphs")

        self.fingerprint_graphs = fingerprint_graphs
        self.graph_metadata = graph_metadata

        return fingerprint_graphs
    
    def create_graph_pairs(self, num_impostor_per_genuine=1):
        """
        Create genuine and impostor pairs from graphs.

        Returns:
            List of tuples: (graph1, graph2, label)
            label = 1 for genuine, 0 for impostor
        """
        if not hasattr(self, 'fingerprint_graphs'):
            raise ValueError("Run graph_maker() first!")

        genuine_pairs = []
        impostor_pairs = []

        # Group graphs by finger (same user, hand, finger_idx)
        finger_groups = {}
        for graph_info in self.fingerprint_graphs:
            key = (graph_info['user_id'], graph_info['hand'], graph_info['finger_idx'])
            if key not in finger_groups:
                finger_groups[key] = []
            finger_groups[key].append(graph_info)

        print("Creating genuine pairs...")
        # Genuine pairs: different impressions of same finger
        for key, graphs in tqdm(finger_groups.items()):
            if len(graphs) < 2:
                continue
            for i in range(len(graphs)):
                for j in range(i + 1, len(graphs)):
                    genuine_pairs.append((
                        graphs[i]['graph'],
                        graphs[j]['graph'],
                        1  # label = 1 for genuine
                    ))

        print("Creating impostor pairs...")
        # Impostor pairs: same finger position, different users
        finger_position_groups = {}
        for graph_info in self.fingerprint_graphs:
            key = (graph_info['hand'], graph_info['finger_idx'])
            if key not in finger_position_groups:
                finger_position_groups[key] = {}

            user_id = graph_info['user_id']
            if user_id not in finger_position_groups[key]:
                finger_position_groups[key][user_id] = []
            finger_position_groups[key][user_id].append(graph_info)

        # Create impostor pairs
        for position_key, users_dict in finger_position_groups.items():
            user_ids = list(users_dict.keys())
            for i in range(len(user_ids)):
                for j in range(i + 1, len(user_ids)):
                    user1_graphs = users_dict[user_ids[i]]
                    user2_graphs = users_dict[user_ids[j]]

                    # Sample impostor pairs
                    for g1 in user1_graphs:
                        for g2 in user2_graphs:
                            impostor_pairs.append((
                                g1['graph'],
                                g2['graph'],
                                0  # label = 0 for impostor
                            ))

        # Balance dataset
        if len(impostor_pairs) > num_impostor_per_genuine * len(genuine_pairs):
            impostor_pairs = random.sample(
                impostor_pairs, 
                num_impostor_per_genuine * len(genuine_pairs)
            )

        all_pairs = genuine_pairs + impostor_pairs
        random.shuffle(all_pairs)

        print(f"Created {len(genuine_pairs)} genuine pairs")
        print(f"Created {len(impostor_pairs)} impostor pairs")
        print(f"Total: {len(all_pairs)} pairs")

        self.split_pairs_train_val_test(all_pairs)
        return all_pairs
    
    def split_pairs_train_val_test(self, all_pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split pairs into train, validation, and test sets.

        Args:
            all_pairs: List of (graph1, graph2, label) tuples
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing

        Returns:
            train_pairs, val_pairs, test_pairs
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        n = len(all_pairs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        random.shuffle(all_pairs)

        train_pairs = all_pairs[:n_train]
        val_pairs = all_pairs[n_train:n_train + n_val]
        test_pairs = all_pairs[n_train + n_val:]

        print(f"\nDataset Split:")
        print(f"  Training:   {len(train_pairs)} pairs ({len(train_pairs)/n*100:.1f}%)")
        print(f"  Validation: {len(val_pairs)} pairs ({len(val_pairs)/n*100:.1f}%)")
        print(f"  Testing:    {len(test_pairs)} pairs ({len(test_pairs)/n*100:.1f}%)")

        return train_pairs, val_pairs, test_pairs





users=load_users_dictionary('processed_minutiae_data.pkl',True)


ko=MinutiaeROCAnalyzer(users)
ko.k_nearest_negihbors(k=3)
ko.graph_maker()
ko.create_graph_pairs()
#ko.genuine_pairs_and_impostor_pairs()

