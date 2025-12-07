# --- xgboost_feature_extractor.py ---

import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_graph_summary_features(graph):
    """
    Computes a final, highly descriptive feature vector.
    
    This NEW version:
    1.  REMOVES the dominant global average angle features.
    2.  ADDS robust local distance features.
    3.  ADDS powerful local RELATIVE ANGLE features.
    """
    # ... (Part 1, 2, and 3 are exactly the same as your last version) ...
    # --- Part 1: Existing Statistical Features (Unchanged) ---
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges // 2
    avg_degree = float(num_edges) / num_nodes if num_nodes > 0 else 0.0
    basic_stats = np.array([num_nodes, num_edges, avg_degree])
    feature_indices_to_use = [0, 1, 2, 3, 6, 7, 8]
    node_features_mean = graph.x[:, feature_indices_to_use].mean(axis=0).numpy()
    node_features_std = graph.x[:, feature_indices_to_use].std(axis=0).numpy()
    if num_nodes > 0:
        dist_to_core = graph.x[:, 6].numpy()
        dist_to_core_stats = np.array([np.median(dist_to_core), np.percentile(dist_to_core, 25), np.percentile(dist_to_core, 75)])
    else:
        dist_to_core_stats = np.zeros(3)
    if graph.num_edges > 0:
        edge_distances = graph.edge_attr[:, 0].numpy()
        edge_angles = graph.edge_attr[:, 1].numpy()
        edge_dist_stats = np.array([edge_distances.mean(), edge_distances.std(), np.median(edge_distances),
                                   np.percentile(edge_distances, 25), np.percentile(edge_distances, 75)])
        dist_hist, _ = np.histogram(edge_distances, bins=5, range=(0, 250))
        edge_angle_stats = np.array([edge_angles.mean(), edge_angles.std(), np.median(edge_angles)])
    else:
        edge_dist_stats = np.zeros(5)
        dist_hist = np.zeros(5)
        edge_angle_stats = np.zeros(3)
    # --- PART 2: MINUTIAE TYPE FEATURES (Unchanged) ---
    if num_nodes > 0:
        minutiae_types = graph.x[:, -1].numpy()
        num_endings = np.sum(minutiae_types == 0)
        num_bifurcations = np.sum(minutiae_types == 1)
        bifurcation_ratio = num_bifurcations / num_nodes if num_nodes > 0 else 0.0
        type_stats = np.array([num_endings, num_bifurcations, bifurcation_ratio])
    else:
        type_stats = np.zeros(3)
    # --- PART 3: RADIAL PROFILING FEATURES (Unchanged) ---
    if num_nodes > 0:
        bins = [0, 40, 80, 120, 1000]
        distances_from_core = graph.x[:, 6].numpy()
        minutiae_sins = graph.x[:, 4].numpy()
        minutiae_cos = graph.x[:, 5].numpy()
        minutiae_angles = np.arctan2(minutiae_sins, minutiae_cos)
        binned_indices = np.digitize(distances_from_core, bins)
        radial_features = []
        for i in range(1, len(bins)):
            angles_in_bin = minutiae_angles[binned_indices == i]
            if len(angles_in_bin) > 0:
                mean_sin = np.mean(np.sin(angles_in_bin))
                mean_cos = np.mean(np.cos(angles_in_bin))
                std_dev_angle = np.std(angles_in_bin)
                radial_features.extend([mean_sin, mean_cos, std_dev_angle])
            else:
                radial_features.extend([0, 0, 0])
        radial_profile_stats = np.array(radial_features)
    else:
        radial_profile_stats = np.zeros(12)
        
    # --- PART 4: LOCAL NEIGHBORHOOD DISTANCE FEATURES (Unchanged) ---
    local_distance_features = []
    if num_nodes > 3:
        coords = graph.x[:, :2].numpy()
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords) # We need indices now
        neighbor_distances = distances[:, 1:]
        mean_local_dist = neighbor_distances.mean()
        std_local_dist = neighbor_distances.std()
        median_local_dist = np.median(neighbor_distances)
        p25_local_dist = np.percentile(neighbor_distances, 25)
        p75_local_dist = np.percentile(neighbor_distances, 75)
        local_distance_features.extend([mean_local_dist, std_local_dist, median_local_dist, p25_local_dist, p75_local_dist])
    else:
        local_distance_features.extend([0, 0, 0, 0, 0])
    local_distance_stats = np.array(local_distance_features)

    # ##########################################################################
    # ### --- PART 5: NEW - LOCAL NEIGHBORHOOD RELATIVE ANGLE FEATURES --- ###
    # ##########################################################################
    local_angle_features = []
    if num_nodes > 3:
        # We can reuse the `indices` from the neighbor search above.
        # Get the angles of all minutiae (already in radians).
        all_angles = np.arctan2(graph.x[:, 4].numpy(), graph.x[:, 5].numpy())
        
        # Get the angles of the neighbors (excluding self)
        neighbor_indices = indices[:, 1:]
        neighbor_angles = all_angles[neighbor_indices]
        
        # Get the angle of each minutia and reshape it for broadcasting
        source_angles = np.expand_dims(all_angles, axis=1)

        # Calculate the difference. This will be an array where each entry is
        # the difference between a minutia's angle and its neighbor's angle.
        relative_angles = neighbor_angles - source_angles

        # Handle angle wrapping (e.g., diff between 355 deg and 5 deg is 10, not -350)
        # We do this by mapping the angles to the range [-pi, pi]
        relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi

        # Now, calculate statistics on these relative angles
        mean_relative_angle = np.mean(relative_angles)
        std_relative_angle = np.std(relative_angles)
        
        local_angle_features.extend([mean_relative_angle, std_relative_angle])
    else:
        local_angle_features.extend([0, 0])

    local_angle_stats = np.array(local_angle_features)

    # --- Concatenate ALL features for the final vector ---
    return np.concatenate([
        basic_stats,
        node_features_mean,
        node_features_std,
        dist_to_core_stats,
        edge_dist_stats,
        dist_hist,
        edge_angle_stats,
        type_stats,
        radial_profile_stats,
        local_distance_stats,
        local_angle_stats # Added our new relative angle features
    ])

def create_feature_vector_for_pair(graph1, graph2):
    features_g1 = get_graph_summary_features(graph1)
    features_g2 = get_graph_summary_features(graph2)
    diff = np.abs(features_g1 - features_g2)
    prod = features_g1 * features_g2
    return np.concatenate([diff, prod])