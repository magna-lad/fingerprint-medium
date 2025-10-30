import numpy as np
import torch
import random
from tqdm import tqdm
from torch_geometric.data import Data
from scipy.spatial import Delaunay

class GraphMinutiae:    
    """
    Handles the conversion of fingerprint minutiae data into graph representations
    and ensures proper, leak-free splitting of the data.
    """
    def __init__(self, users_minutiae):
        self.users_minutiae = users_minutiae
        self.fingerprint_graphs = []
        self.graph_metadata = {}

    @staticmethod
    def old_normalize_minutiae_features(minutiae):
        """Normalizes minutiae features into a standard format."""
        coords = minutiae[:, :2].astype(np.float32)
        center = coords.mean(axis=0)
        std = coords.std(axis=0) + 1e-8
        norm_coords = (coords - center) / std
        
        type_col = minutiae[:, 2].astype(int)
        type_onehot = np.eye(2)[type_col]

        angle_rad = np.deg2rad(minutiae[:, 3])
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)

        return np.column_stack([norm_coords, type_onehot, angle_sin, angle_cos])
    
    @staticmethod
    def normalize_minutiae_features(minutiae):
        """
        Normalizes minutiae features by centering them relative to a core point proxy.
        This version avoids per-fingerprint scaling, providing a consistent geometric
        representation for the model to learn from.
        """
        coords = minutiae[:, :2].astype(np.float32)
    
        # --- NEW: Use the core point proxy as the origin (0,0) ---
        core_point = GraphMinutiae.find_core_proxy(minutiae)
        centered_coords = coords - core_point
    
        # Minutiae type (one-hot encoded)
        type_col = minutiae[:, 2].astype(int)
        # Ensure it handles cases with only one type of minutia
        type_onehot = np.zeros((len(type_col), 2), dtype=np.float32)
        type_onehot[np.arange(len(type_col)), type_col] = 1.0
    
        # Minutiae's own angle (sin/cos encoded)
        angle_rad = np.deg2rad(minutiae[:, 3])
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)
    
        # --- NEW GLOBAL FEATURES (retained from your code, which is good) ---
    
        # 1. Distance from each minutia to the new origin (the core)
        # This is now a more meaningful feature since the coordinates are centered.
        # We don't normalize it per-fingerprint anymore.
        dist_to_core = np.linalg.norm(centered_coords, axis=1)
        dist_to_core = dist_to_core.reshape(-1, 1)
    
        # 2. Angle from each minutia TO the core
        deltas = core_point - coords # Or -centered_coords
        angle_to_core_rad = np.arctan2(deltas[:, 1], deltas[:, 0])
        core_angle_sin = np.sin(angle_to_core_rad).reshape(-1, 1)
        core_angle_cos = np.cos(angle_to_core_rad).reshape(-1, 1)
        
        # --- Final feature vector (still 9 features, but more stable) ---
        return np.column_stack([
            centered_coords,    # features 0, 1 (NOT scaled per-print)
            type_onehot,        # features 2, 3
            angle_sin,          # feature 4
            angle_cos,          # feature 5
            dist_to_core,       # feature 6 (NOT scaled per-print)
            core_angle_sin,     # feature 7
            core_angle_cos      # feature 8
        ])

    
    def _build_single_graph(self, minutiae, graph_id):
        """Builds a single graph using robust Delaunay Triangulation."""
        if minutiae is None or len(minutiae) < 4:
            return None

        normalized_features = self.normalize_minutiae_features(minutiae)
        coords = normalized_features[:, :2]

        try:
            tri = Delaunay(coords)
        except Exception:
            return None # Fails if all points are co-linear
            
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[0], simplex[2]))))
        
        if not edges: return None

        edge_list = np.array(list(edges))
        edge_sources = np.concatenate([edge_list[:, 0], edge_list[:, 1]])
        edge_targets = np.concatenate([edge_list[:, 1], edge_list[:, 0]])
        edge_index = torch.from_numpy(np.array([edge_sources, edge_targets])).long()
        
        src_coords, tgt_coords = coords[edge_sources], coords[edge_targets]
        distances = np.linalg.norm(src_coords - tgt_coords, axis=1)
        src_sin, src_cos = normalized_features[edge_sources, 4], normalized_features[edge_sources, 5]
        tgt_sin, tgt_cos = normalized_features[edge_targets, 4], normalized_features[edge_targets, 5]
        dot_product = np.clip((src_cos * tgt_cos) + (src_sin * tgt_sin), -1.0, 1.0)
        relative_angles = np.arccos(dot_product)
        edge_features = np.vstack([distances, relative_angles]).T
    
        edge_attr = torch.from_numpy(edge_features).float()
        x = torch.from_numpy(normalized_features).float()
    
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.graph_id = graph_id
        return graph
    

    @staticmethod
    def augment_minutiae(minutiae, max_rotation=30, max_translation=25):
        """
        Applies random rotation and translation to a set of minutiae.
        """
        angle_deg = np.random.uniform(-max_rotation, max_rotation)
        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[c, -s], [s, c]])
        coords = minutiae[:, :2]
        center = coords.mean(axis=0)
        rotated_coords = (coords - center) @ rotation_matrix.T + center
        translation = np.random.uniform(-max_translation, max_translation, size=2)
        final_coords = rotated_coords + translation
        final_angles = (minutiae[:, 3] + angle_deg) % 360
        augmented_minutiae = minutiae.copy()
        augmented_minutiae[:, :2] = final_coords
        augmented_minutiae[:, 3] = final_angles
        return augmented_minutiae
    
    @staticmethod
    def find_core_proxy(minutiae):
        """Calculates the center of mass of minutiae as a proxy for the core point."""
        return minutiae[:, :2].mean(axis=0)

    def graph_maker(self):
        """Iterates through all fingerprints and builds a list of graph objects."""
        print("Building graphs from all fingerprint minutiae...")
        graphs_with_meta = []
        for uid, udata in tqdm(self.users_minutiae.items(), desc="Processing Users"):
            for hand, fingers in udata['fingers'].items():
                for finger_idx, impressions in enumerate(fingers):
                    for impr_idx, impression_data in enumerate(impressions):
                        graph_id = f"{uid}_{hand}_{finger_idx}_{impr_idx}"
                        minutiae = impression_data["minutiae"]
                        graph = self._build_single_graph(minutiae, graph_id)
                        if graph is not None:
                            meta_info = {'graph': graph, 'user_id': uid, 'hand': hand, 
                                         'finger_idx': finger_idx, 'impression_idx': impr_idx, 'graph_id': graph_id}
                            graphs_with_meta.append(meta_info)
        
        self.fingerprint_graphs = graphs_with_meta
        self.graph_metadata = {info['graph_id']: info for info in graphs_with_meta}
        print(f"Successfully built {len(self.fingerprint_graphs)} graphs.")
        return self.fingerprint_graphs

    def create_graph_pairs(self, num_impostors_per_genuine=1):
        """Creates genuine and impostor pairs for evaluation."""
        genuine_pairs, impostor_pairs = [], []
        finger_groups = {}
        for info in self.fingerprint_graphs:
            key = (info['user_id'], info['hand'], info['finger_idx'])
            finger_groups.setdefault(key, []).append(info['graph'])
        
        print("Creating genuine pairs...")
        for key, graphs in tqdm(finger_groups.items()):
            if len(graphs) >= 2:
                for i in range(len(graphs)):
                    for j in range(i + 1, len(graphs)):
                        genuine_pairs.append((graphs[i], graphs[j], 1))

        finger_pos_groups = {}
        for info in self.fingerprint_graphs:
            key = (info['hand'], info['finger_idx'])
            finger_pos_groups.setdefault(key, []).append(info['graph'])
        
        print("Creating impostor pairs...")
        for key, graphs in tqdm(finger_pos_groups.items()):
            user_ids = {self.graph_metadata[g.graph_id]['user_id'] for g in graphs}
            if len(user_ids) >= 2:
                for i in range(len(graphs)):
                    for j in range(i + 1, len(graphs)):
                        uid1 = self.graph_metadata[graphs[i].graph_id]['user_id']
                        uid2 = self.graph_metadata[graphs[j].graph_id]['user_id']
                        if uid1 != uid2:
                            impostor_pairs.append((graphs[i], graphs[j], 0))
        
        if impostor_pairs and len(impostor_pairs) > num_impostors_per_genuine * len(genuine_pairs):
            impostor_pairs = random.sample(impostor_pairs, num_impostors_per_genuine * len(genuine_pairs))
            
        all_pairs = genuine_pairs + impostor_pairs
        random.shuffle(all_pairs)
        print(f"Created {len(genuine_pairs)} genuine pairs and {len(impostor_pairs)} impostor pairs.")
        return all_pairs

    def get_user_splits(self, train_ratio=0.7, val_ratio=0.15):
        """Splits USER IDs into disjoint train, validation, and test sets."""
        all_user_ids = sorted(list(self.users_minutiae.keys()))
        random.shuffle(all_user_ids)
        n_users = len(all_user_ids)
        n_train, n_val = int(n_users * train_ratio), int(n_users * val_ratio)
        train_users = set(all_user_ids[:n_train])
        val_users = set(all_user_ids[n_train : n_train + n_val])
        test_users = set(all_user_ids[n_train + n_val:])
        print(f"\nUser Split: {len(train_users)} train, {len(val_users)} validation, {len(test_users)} test.")
        return train_users, val_users, test_users

    def split_pairs_by_user(self, all_pairs, train_users, val_users, test_users):
        """Assigns a list of pairs to splits based on user IDs."""
        train_pairs, val_pairs, test_pairs = [], [], []
        for g1, g2, label in all_pairs:
            uid1 = self.graph_metadata[g1.graph_id]['user_id']
            uid2 = self.graph_metadata[g2.graph_id]['user_id']
            if uid1 in train_users and uid2 in train_users: train_pairs.append((g1, g2, label))
            elif uid1 in val_users and uid2 in val_users: val_pairs.append((g1, g2, label))
            elif uid1 in test_users and uid2 in test_users: test_pairs.append((g1, g2, label))
        print(f"Pair Split: {len(train_pairs)} train, {len(val_pairs)} validation, {len(test_pairs)} test.")
        return train_pairs, val_pairs, test_pairs