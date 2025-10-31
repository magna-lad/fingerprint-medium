# --- xgboost_feature_extractor.py ---

import numpy as np

def get_graph_summary_features(graph):
    """
    Computes a final, highly descriptive feature vector.
    This version includes RADIAL PROFILING features and NEW minutiae type statistics.
    """
    # --- Part 1: Existing Statistical Features ---
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges // 2
    avg_degree = float(num_edges) / num_nodes if num_nodes > 0 else 0.0
    basic_stats = np.array([num_nodes, num_edges, avg_degree])

    node_features_mean = graph.x[:, :-1].mean(axis=0).numpy() # Exclude our new type column
    node_features_std = graph.x[:, :-1].std(axis=0).numpy()

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

    # --- PART 2: NEW - MINUTIAE TYPE FEATURES ---
    if num_nodes > 0:
        # The last column now holds the original minutiae type (0 for ending, 1 for bifurcation)
        minutiae_types = graph.x[:, -1].numpy()
        num_endings = np.sum(minutiae_types == 0)
        num_bifurcations = np.sum(minutiae_types == 1)
        # Ratio is a powerful feature as it's independent of the total number of minutiae
        bifurcation_ratio = num_bifurcations / num_nodes if num_nodes > 0 else 0.0
        type_stats = np.array([num_endings, num_bifurcations, bifurcation_ratio])
    else:
        type_stats = np.zeros(3)


    # --- PART 3: RADIAL PROFILING FEATURES ---
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


    # --- Concatenate ALL features for the final vector ---
    return np.concatenate([
        basic_stats,
        node_features_mean,
        node_features_std,
        dist_to_core_stats,
        edge_dist_stats,
        dist_hist,
        edge_angle_stats,
        type_stats, # Added our new features
        radial_profile_stats
    ])

def create_feature_vector_for_pair(graph1, graph2):
    features_g1 = get_graph_summary_features(graph1)
    features_g2 = get_graph_summary_features(graph2)
    # Using both difference and element-wise product can sometimes capture interactions better
    diff = np.abs(features_g1 - features_g2)
    prod = features_g1 * features_g2
    return np.concatenate([diff, prod])