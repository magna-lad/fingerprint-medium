import numpy as np

def get_graph_summary_features(graph):
    """
    Computes a single, richer feature vector summarizing the properties of one graph.
    """
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    avg_degree = float(num_edges) / num_nodes if num_nodes > 0 else 0.0
    basic_stats = np.array([num_nodes, num_edges, avg_degree])

    node_features_mean = graph.x.mean(axis=0).numpy()
    node_features_std = graph.x.std(axis=0).numpy()
    
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

    return np.concatenate([basic_stats, node_features_mean, node_features_std, dist_to_core_stats,
                           edge_dist_stats, dist_hist, edge_angle_stats])

def create_feature_vector_for_pair(graph1, graph2):
    """
    Creates a final feature vector for a pair of graphs by taking the absolute difference.
    """
    features_g1 = get_graph_summary_features(graph1)
    features_g2 = get_graph_summary_features(graph2)
    return np.abs(features_g1 - features_g2)