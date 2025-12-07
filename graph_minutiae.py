import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import Delaunay
import tqdm
import random
from sklearn.neighbors import kneighbors_graph

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
    def normalize_minutiae_features(minutiae,orientation_map):
        coords = minutiae[:, :2].astype(np.float32)
        # --- IMPROVEMENT HOOK ---
        # A true core point detected via Poincaré Index would replace find_core_proxy.
        # For example: core_point = find_true_core(orientation_map)
        # This would make the 'centered_coords' and 'dist_to_core' features far more robust.
        core_point = GraphMinutiae.find_core_proxy(minutiae)
        core_point = GraphMinutiae.find_true_core(orientation_map)
        if core_point is None:
            core_point = GraphMinutiae.find_core_proxy(minutiae)
        centered_coords = coords - core_point
        type_col = minutiae[:, 2].astype(int)
        type_onehot = np.zeros((len(type_col), 2), dtype=np.float32)
        type_onehot[np.arange(len(type_col)), type_col] = 1.0
        angle_rad = np.deg2rad(minutiae[:, 3])
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)
        dist_to_core = np.linalg.norm(centered_coords, axis=1).reshape(-1, 1)
        deltas = core_point - coords
        angle_to_core_rad = np.arctan2(deltas[:, 1], deltas[:, 0])
        core_angle_sin = np.sin(angle_to_core_rad).reshape(-1, 1)
        core_angle_cos = np.cos(angle_to_core_rad).reshape(-1, 1)
        # We also need to keep the original minutiae type for new features.
        # So we add the type column at the end for easy access.
        return np.column_stack([centered_coords, type_onehot, angle_sin, angle_cos, dist_to_core, core_angle_sin, core_angle_cos, minutiae[:, 2].astype(np.float32)])


    def _build_single_graph(self, minutiae, orientation_map, graph_id):
        """Builds a single graph using robust Delaunay Triangulation."""
        MINUTIAE_THRESHOLD = 20
        if minutiae is None or len(minutiae) < MINUTIAE_THRESHOLD:
            return None # Discard low-quality graph
        if minutiae is None or len(minutiae) < 4:
            return None
        
        normalized_features = self.normalize_minutiae_features(minutiae,orientation_map)
        coords = normalized_features[:, :2]

        try:
            tri = Delaunay(coords)
        except Exception:
            return None
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[0], simplex[2]))))
        if not edges: return None
        edge_list = np.array(list(edges))
        edge_sources, edge_targets = edge_list[:, 0], edge_list[:, 1]
        src_coords, tgt_coords = coords[edge_sources], coords[edge_targets]
        distances = np.linalg.norm(src_coords - tgt_coords, axis=1)
        src_sin, src_cos = normalized_features[edge_sources, 4], normalized_features[edge_sources, 5]
        tgt_sin, tgt_cos = normalized_features[edge_targets, 4], normalized_features[edge_targets, 5]
        dot_product = np.clip((src_cos * tgt_cos) + (src_sin * tgt_sin), -1.0, 1.0)
        relative_angles = np.arccos(dot_product)
        deltas = tgt_coords - src_coords
        edge_orientations = np.arctan2(deltas[:, 1], deltas[:, 0])
        edge_features = np.vstack([distances, relative_angles, edge_orientations]).T
        full_edge_sources = np.concatenate([edge_list[:, 0], edge_list[:, 1]])
        full_edge_targets = np.concatenate([edge_list[:, 1], edge_list[:, 0]])
        edge_index = torch.from_numpy(np.array([full_edge_sources, full_edge_targets])).long()
        edge_attr = torch.from_numpy(np.concatenate([edge_features, edge_features])).float()
        x = torch.from_numpy(normalized_features).float()
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.graph_id = graph_id
        # Store raw minutiae in graph object for potential augmentation later
        graph.raw_minutiae = minutiae
        graph.orientation_map = orientation_map
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
        for uid, udata in tqdm.tqdm(self.users_minutiae.items(), desc="Processing Users"):
            for hand, fingers in udata['fingers'].items():
                for finger_idx, impressions in enumerate(fingers):
                    for impr_idx, impression_data in enumerate(impressions):
                        graph_id = f"{uid}_{hand}_{finger_idx}_{impr_idx}"
                        minutiae = impression_data["minutiae"]
                        orientation_map = impression_data["orientation_map"] 
                        graph = self._build_single_graph(minutiae, orientation_map, graph_id)
                        if graph is not None:
                            meta_info = {'graph': graph, 'user_id': uid, 'hand': hand,
                                         'finger_idx': finger_idx, 'impression_idx': impr_idx, 'graph_id': graph_id}
                            graphs_with_meta.append(meta_info)
        self.fingerprint_graphs = graphs_with_meta
        self.graph_metadata = {info['graph_id']: info for info in graphs_with_meta}
        print(f"Successfully built {len(self.fingerprint_graphs)} graphs.")
        return self.fingerprint_graphs

    def create_graph_pairs(self, num_impostors_per_genuine=3): # Increased default ratio
        """Creates genuine and impostor pairs for evaluation."""
        genuine_pairs, impostor_pairs = [], []
        finger_groups = {}
        for info in self.fingerprint_graphs:
            key = (info['user_id'], info['hand'], info['finger_idx'])
            finger_groups.setdefault(key, []).append(info['graph'])
        print("Creating genuine pairs...")
        for key, graphs in tqdm.tqdm(finger_groups.items()):
            if len(graphs) >= 2:
                for i in range(len(graphs)):
                    for j in range(i + 1, len(graphs)):
                        genuine_pairs.append((graphs[i], graphs[j], 1))
        finger_pos_groups = {}
        for info in self.fingerprint_graphs:
            key = (info['hand'], info['finger_idx'])
            finger_pos_groups.setdefault(key, []).append(info['graph'])
        print("Creating impostor pairs...")
        all_graphs = [info['graph'] for info in self.fingerprint_graphs]
        target_num_impostors = num_impostors_per_genuine * len(genuine_pairs)

        while len(impostor_pairs) < target_num_impostors:
            g1, g2 = random.sample(all_graphs, 2)
            uid1 = self.graph_metadata[g1.graph_id]['user_id']
            uid2 = self.graph_metadata[g2.graph_id]['user_id']
            if uid1 != uid2:
                impostor_pairs.append((g1, g2, 0))

        all_pairs = genuine_pairs + impostor_pairs
        random.shuffle(all_pairs)
        print(f"Created {len(genuine_pairs)} genuine pairs and {len(impostor_pairs)} impostor pairs.")
        return all_pairs

    def get_user_splits(self, train_ratio=0.7, val_ratio=0.15):
        """Splits USER IDs into disjoint train, validation, and test sets."""
        all_user_ids = sorted(list(self.users_minutiae.keys()))
        random.seed(42) # for reproducibility
        random.shuffle(all_user_ids)
        n_users = len(all_user_ids)
        n_train = int(n_users * train_ratio)
        n_val = int(n_users * val_ratio)
        train_users = set(all_user_ids[:n_train])
        val_users = set(all_user_ids[n_train:n_train + n_val])
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
    

    # Place this inside the GraphMinutiae class in graph_minutiae.py

    @staticmethod
    def find_true_core(orientation_map, block_size=16):
        """
        Calculates the location of the core point using the Poincaré Index.

        This method iterates through an orientation map, calculates the Poincaré
        Index for each point, and identifies singularities (cores and deltas).

        Input:
            orientation_map: The 2D numpy array from skeleton_maker (self.angle_gabor).
                             Angles are expected to be in radians.
            block_size: The pixel width/height of a single block in the orientation map,
                        used to scale the coordinates back to the original image size.

        Output:
            A tuple (x, y) of the most likely core's coordinates in the original
            image space, or None if no core is found.
        """
        # Import numpy here if it's not at the top of the file
        import numpy as np

        # The orientation map values are doubled for Poincaré calculation to handle the 180-degree ambiguity
        doubled_orientation = 2 * orientation_map

        rows, cols = doubled_orientation.shape

        # Store found singularities
        cores = []
        deltas = []

        # Iterate through the interior of the map (excluding borders)
        # because we need a 3x3 window around each point.
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):

                # Define the 8-neighbor path around the current cell (r, c) in clockwise order
                path = [
                    doubled_orientation[r - 1, c - 1], doubled_orientation[r - 1, c],
                    doubled_orientation[r - 1, c + 1], doubled_orientation[r, c + 1],
                    doubled_orientation[r + 1, c + 1], doubled_orientation[r + 1, c],
                    doubled_orientation[r + 1, c - 1], doubled_orientation[r, c - 1],
                ]

                # --- Calculate the Poincaré Index by summing angle differences ---
                index_sum = 0.0
                for k in range(8):
                    # Get the angle difference between the current and next point on the path
                    # The modulo operator (%) closes the loop, connecting the last point back to the first.
                    diff = path[(k + 1) % 8] - path[k]

                    # --- Handle Angle Wrapping ---
                    # This is the most critical step. A simple subtraction is not enough for circular data.
                    # We need to find the shortest path between angles.
                    if diff > np.pi:
                        diff -= 2 * np.pi
                    elif diff < -np.pi:
                        diff += 2 * np.pi

                    index_sum += diff

                # --- Classify the Singularity ---
                # Divide by 2*pi to get the index. We check if it's close to the theoretical values.
                # Index is ~0.5 for a core, ~-0.5 for a delta, ~0 for a normal point.
                # A tolerance is needed for discrete data.
                tolerance = 0.1 

                # Core: Poincaré index is +1/2 (sum is pi)
                if 0.5 - tolerance < (index_sum / (2 * np.pi)) < 0.5 + tolerance:
                    cores.append((r, c))

                # Delta: Poincaré index is -1/2 (sum is -pi)
                elif -0.5 - tolerance < (index_sum / (2 * np.pi)) < -0.5 + tolerance:
                    deltas.append((r, c))

        # --- Select the Best Core ---
        if not cores:
            return None # No cores found

        # If multiple cores are found (e.g., in an arch or due to noise), a common
        # heuristic is to choose the "lowest" one (highest row index), as this is
        # often the most dominant and stable core.
        cores.sort(key=lambda item: item[0], reverse=True)
        best_core_r, best_core_c = cores[0]

        # --- Scale Coordinates ---
        # Convert the core's location from the small orientation map grid back
        # to the full-size image coordinate system. We add half a block size
        # to center the coordinate within its block.
        scaled_x = best_core_c * block_size + (block_size // 2)
        scaled_y = best_core_r * block_size + (block_size // 2)

        return (scaled_x, scaled_y)