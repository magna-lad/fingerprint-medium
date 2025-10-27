import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from load_save import *
from itertools import combinations, product
import random
import torch
from torch_geometric.data import Data
    
class GraphMinutiae:
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

    # find the neighbouring minutiae for a minutiae and make feature vectors to feed in the model
    
        '''
        
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
        Extract k nearest neighbors for each minutia point based on Euclidean distance of normalised (x,y) coordinates.
    
        Parameters:
        minutiae: np.array of shape (N, 4) where columns represent [x, y, type, angle]
        k: int - number of nearest neighbors
    
        Returns:
        neighbors_indices: list of lists - each list contains indices of nearest neighbors for corresponding minutia
        neighbors_distances: list of lists - distances corresponding to neighbors_indices
        """
        
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
    def compute_relative_angles_normalized(normalized_minutiae, neighbors_indices):
        """
        Compute relative angle differences between each minutia and its neighbors
        using normalized [sin, cos] angle representations.
    
        Parameters:
        normalized_minutiae: np.array of shape (N, 6) - [x, y, t0, t1, sin(a), cos(a)], a- angle to be found
        neighbors_indices: list of lists - neighbors for each minutia
    
        Returns:
        neighbors_relative_angles: list of lists - relative angle differences in radians (or degrees, whichever you prefer)
        """
        neighbors_relative_angles = []
        
        
        angles_sin = normalized_minutiae[:, 4]
        angles_cos = normalized_minutiae[:, 5]

        for i, neighbors in enumerate(neighbors_indices): # i- self, neigbhors- neighbours of self minutiae
            # Reference angle components
            ref_sin = angles_sin[i]
            ref_cos = angles_cos[i]
            
            rel_angles = []
            for nbr_idx in neighbors:
                # Neighbor angle components
                nbr_sin = angles_sin[nbr_idx]
                nbr_cos = angles_cos[nbr_idx]
                
                 
                # cos(a_ref)cos(a_nbr) + sin(a_ref)sin(a_nbr) = cos(a_ref - a_nbr)
                product = (ref_cos * nbr_cos) + (ref_sin * nbr_sin)
                
                # Clip for numerical stability (e.g., 1.0000001)
                product = np.clip(product, -1.0, 1.0)
                
                # Get angle in radians from acos
                rel_angle_rad = np.arccos(product)
                
                # Convert to degrees, if required, my pipeline uses radians further down
                #rel_angle_deg = np.degrees(rel_angle_rad)
                
                rel_angles.append(rel_angle_rad)
                
            neighbors_relative_angles.append(rel_angles)
            
        return neighbors_relative_angles
    

    @staticmethod
    def create_feature_vectors(minutiae, neighbors_indices, neighbors_distances, neighbors_relative_angles):
        """
        Create feature vectors for each minutia combining own features and neighbors' distances and relative angle differences.
    
        Parameters:
        minutiae: np.array of shape (N, 4) - columns: [x, y, type, angle]
        neighbors_indices: list of lists [self,neighoburs]
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
    
    @staticmethod
    def augment_minutiae(minutiae, max_rotation=15, max_translation=10):
        """
        Apply random rotation and translation to a set of minutiae.
        minutiae: np.array of shape [N, 4]
        """
        # 1. Random Rotation
        angle_deg = np.random.uniform(-max_rotation, max_rotation)
        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[c, -s], [s, c]])

        coords = minutiae[:, :2]
        center = coords.mean(axis=0)
        rotated_coords = (coords - center) @ rotation_matrix.T + center

        # 2. Random Translation
        translation = np.random.uniform(-max_translation, max_translation, size=2)
        final_coords = rotated_coords + translation

        # 3. Update Angles
        final_angles = (minutiae[:, 3] + angle_deg) % 360

        augmented_minutiae = minutiae.copy()
        augmented_minutiae[:, :2] = final_coords
        augmented_minutiae[:, 3] = final_angles

        return augmented_minutiae
    
    
    def get_user_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        to sort on the basis of user ids
        """
        all_user_ids = sorted(list(self.users_minutiae.keys()))
        random.shuffle(all_user_ids)
        n = len(all_user_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_users = set(all_user_ids[:n_train])
        val_users = set(all_user_ids[n_train:n_train + n_val])
        test_users = set(all_user_ids[n_train + n_val:])
        return train_users, val_users, test_users

    # revisit- done ok
    def split_pairs_by_user(self, all_pairs, train_users, val_users, test_users):
        graph_to_user = {info['graph_id']: info['user_id'] for info in self.fingerprint_graphs}
        train_pairs, val_pairs, test_pairs = [], [], []
        for g1, g2, label in all_pairs:
            uid1 = graph_to_user[g1.graph_id]
            uid2 = graph_to_user[g2.graph_id]
            if uid1 in train_users and uid2 in train_users:
                train_pairs.append((g1, g2, label))
            elif uid1 in val_users and uid2 in val_users:
                val_pairs.append((g1, g2, label))
            elif uid1 in test_users and uid2 in test_users:
                test_pairs.append((g1, g2, label))
        return train_pairs, val_pairs, test_pairs
       
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

                            neighbors_indices, neighbors_distances = self.extract_neighbors(image["minutiae"], k=5)
                            neighbors_relative_angles = self.compute_relative_angles(image["minutiae"], neighbors_indices)
                            feature_vectors = self.create_feature_vectors(image["minutiae"], neighbors_indices, neighbors_distances, neighbors_relative_angles)
                            self.users_feature_vectors[user_id]["fingers"][hand][finger_index][impression_index] = feature_vectors
        #print(len(self.users_feature_vectors["000"]["fingers"]["L"][0][1]))
    
     #revisit- done ok
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
    #revisit- done ok

    def create_single_graph(self, minutiae_array, k=5):
        """
        Takes a raw minutiae array and converts it into a single PyG Data object (a graph).
        minutiae_array: np.array of shape [N, 4] -> [x, y, type, angle]
        """
        if minutiae_array is None or len(minutiae_array) < k + 1:
            return None # Not enough minutiae to build a valid graph

        # Normalize the minutiae features (same logic as in graph_maker)
        normalized_features = self.normalize_minutiae_features(minutiae_array)

        # Build KNN structure
        neighbors_indices, neighbors_distances = self.extract_neighbors(normalized_features, k=k)
        neighbors_relative_angles = self.compute_relative_angles_normalized(normalized_features, neighbors_indices)

        # Create the graph Data object for a single fingerprint
        graph = self.build_graph_from_minutiae(
            normalized_features,
            neighbors_indices,
            neighbors_distances,
            neighbors_relative_angles
        )
        return graph

    def normalize_minutiae_features(self,minutiae):
        # minutiae: numpy array of shape [N, 4]: x, y, type, angle (degrees)
        coords = minutiae[:, :2]  # x and y columns
        # handle case where constant or all same coords
        if coords.std(axis=0).min() < 1e-5:
            coords = coords - coords.mean(axis=0)
        else:
            coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
        # angle normalization
        norm_angle = (minutiae[:, 3] % 360) / 360.0
        # normalized features as [x, y, type, angle_norm]
        type_col = minutiae[:, 2].astype(int)
        type_onehot = np.eye(2)[type_col]
        angle_rad = norm_angle * 2 * np.pi
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)
        normalized = np.column_stack([coords, type_onehot, angle_sin, angle_cos])
        return normalized

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
                            # Per impression, before graph creation:
                            minutiae = impression[:, :4]
                            minutiae = self.normalize_minutiae_features(minutiae)
                            
                            # Build KNN structure
                            neighbors_indices, neighbors_distances = self.extract_neighbors(minutiae, k=5)
                            neighbors_relative_angles = self.compute_relative_angles_normalized(minutiae, neighbors_indices)

                            # Create graph
                            graph = self.build_graph_from_minutiae(
                                minutiae,  #[coords, type_onehot, angle_sin, angle_cos]
                                neighbors_indices,
                                neighbors_distances,
                                neighbors_relative_angles
                            )

                            # Create unique identifier for this impression
                            graph_id = f"{uid}_{hand}_{finger_idx}_{impression_idx}"
                            graph.graph_id = graph_id
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
    
    # for FingerprintContrastiveLoss
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
        print(len(genuine_pairs))
        #print(impostor_pairs)
        
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

        #self.split_pairs_train_val_test(all_pairs)
        return all_pairs
    





    # revisit
    # for fingerprintTripletLoss

    def create_triplets(self, train_users):
        """
        Create triplets (anchor, positive, negative) for training.
        Performs on-the-fly augmentation for anchor and positive samples.
        """
        if not hasattr(self, 'fingerprint_graphs'):
            raise ValueError("Run graph_maker() first to have graphs for negatives and validation!")
    
        triplets = []
        
        # 1. Group fingerprints by finger for users in the training set
        finger_groups = {}
        for graph_info in self.fingerprint_graphs:
            user_id = graph_info['user_id']
            if user_id in train_users:
                key = (user_id, graph_info['hand'], graph_info['finger_idx'])
                if key not in finger_groups:
                    finger_groups[key] = []
                # Store the info needed for lookup, not the graph itself
                finger_groups[key].append(graph_info)
        
        # Get a list of pre-built graphs from the training set to serve as negatives
        negative_candidate_graphs = [g['graph'] for g in self.fingerprint_graphs if g['user_id'] in train_users]
    
        print("Creating augmented triplets for training...")
        for key, finger_impressions in tqdm(finger_groups.items(), desc="Generating triplets"):
            if len(finger_impressions) < 2:
                continue
                
            # For every combination of two impressions from the same finger...
            for i in range(len(finger_impressions)):
                for j in range(i + 1, len(finger_impressions)):
                    anchor_info = finger_impressions[i]
                    positive_info = finger_impressions[j]
    
                    # 2. LOOKUP: Get the raw minutiae from the original data structure
                    # We slice with [:4] because k_nearest_neighbors might have added other features

                    # cannot use self.users_feature_vectors- as not normalized
                    raw_anchor_minutiae = self.users_feature_vectors[anchor_info['user_id']]['fingers'][anchor_info['hand']][anchor_info['finger_idx']][anchor_info['impression_idx']][:, :4]
                    
                    raw_positive_minutiae = self.users_feature_vectors[positive_info['user_id']]['fingers'][positive_info['hand']][positive_info['finger_idx']][positive_info['impression_idx']][:, :4]
                    
                    
                    # 3. AUGMENT: Apply random transformations
                    aug_anchor_minutiae = self.augment_minutiae(raw_anchor_minutiae)
                    aug_positive_minutiae = self.augment_minutiae(raw_positive_minutiae)
    
                    # 4. BUILD GRAPH ON-THE-FLY
                    anchor_graph = self.create_single_graph(aug_anchor_minutiae)
                    positive_graph = self.create_single_graph(aug_positive_minutiae)
    
                    # Skip if augmentation resulted in an invalid graph (e.g., too few points)
                    if anchor_graph is None or positive_graph is None:
                        continue
                    
                    # 5. FIND A NEGATIVE
                    # We can use the pre-built graphs for this
                    while True:
                        negative_graph = random.choice(negative_candidate_graphs)
                        # Ensure the negative is from a different finger
                        if negative_graph.graph_id != anchor_info['graph_id'] and negative_graph.graph_id != positive_info['graph_id']:
                            break
                        
                    triplets.append((anchor_graph, positive_graph, negative_graph))
        
        random.shuffle(triplets)
        print(f"Created {len(triplets)} augmented training triplets.")
        return triplets

    
    
    
    def split_pairs_train_val_test(self, all_pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split pairs into train, validation, and test sets based on USER ID
        to prevent data leakage.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # 1. Get all unique user IDs that have at least one graph
        all_user_ids = sorted(list(self.users_feature_vectors.keys()))
        random.shuffle(all_user_ids)

        # 2. Split the USER IDs
        n = len(all_user_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_users = set(all_user_ids[:n_train])
        val_users = set(all_user_ids[n_train:n_train + n_val])
        test_users = set(all_user_ids[n_train + n_val:])
        
        print(f"\nUser Split:")
        print(f"  Training Users:   {len(train_users)}")
        print(f"  Validation Users: {len(val_users)}")
        print(f"  Testing Users:    {len(test_users)}")

        # 3. We need a way to get the user ID from a graph object
        #    This is why 'graph_metadata' is useful. We map graph_id -> user_id
        graph_to_user = {}
        for info in self.fingerprint_graphs:
            graph_to_user[info['graph_id']] = info['user_id']
            
        # 4. Assign pairs to splits
        train_pairs, val_pairs, test_pairs = [], [], []
        
        for g1, g2, label in all_pairs:
            try:
                uid1 = graph_to_user[g1.graph_id]
                uid2 = graph_to_user[g2.graph_id]
            except:
                print("Error: Graph ID not found. Check graph_maker implementation.")
                continue
                    
            if uid1 is None or uid2 is None:
                # This can happen with the hacky method.
                # A proper graph_id on the object is required.
                continue

            # Assign pair based on user IDs
            if uid1 in train_users and uid2 in train_users:
                train_pairs.append((g1, g2, label))
            elif uid1 in val_users and uid2 in val_users:
                val_pairs.append((g1, g2, label))
            elif uid1 in test_users and uid2 in test_users:
                test_pairs.append((g1, g2, label))
            # Cross-split pairs (e.g., train user vs val user) are impostors
            # and can be added to the appropriate split.
            # For simplicity, we only take pairs from *within* the same user split.

        print(f"\nDataset Split (Pairs):")
        n_total = len(train_pairs) + len(val_pairs) + len(test_pairs)
        print(f"  Training:   {len(train_pairs)} pairs ({len(train_pairs)/n_total*100:.1f}%)")
        print(f"  Validation: {len(val_pairs)} pairs ({len(val_pairs)/n_total*100:.1f}%)")
        print(f"  Testing:    {len(test_pairs)} pairs ({len(test_pairs)/n_total*100:.1f}%)")

        return train_pairs, val_pairs, test_pairs
    

    # for contrastive loss

    def old_split_pairs_train_val_test(self, all_pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
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





#users=load_users_dictionary('processed_minutiae_data.pkl',True)
#
#
#ko=GraphMinutiae(users)
#ko.k_nearest_negihbors(k=3)
#ko.graph_maker()
#ko.create_graph_pairs()


