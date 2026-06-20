# =============================================================================
#  Community Detection & Centrality Analysis
# =============================================================================
"""
Graph-theoretic analysis of patient similarity networks.

Key analyses:
    - Louvain community detection → patient risk subgroups
    - Degree centrality → highly connected "archetypal" patients
    - Betweenness centrality → bridge patients between risk communities
    - Closeness centrality → patients central to the overall network
"""

from typing import Dict, Tuple

import networkx as nx
import community as community_louvain  # python-louvain


def detect_communities(
    G: nx.Graph,
    resolution: float = 1.0,
) -> Tuple[Dict, float]:
    """
    Detect patient communities using the Louvain algorithm.

    Parameters
    ----------
    G : nx.Graph
        Patient similarity network.
    resolution : float
        Resolution parameter for Louvain. Higher values produce
        more, smaller communities.

    Returns
    -------
    partition : dict
        Mapping of node → community ID.
    modularity : float
        Modularity score Q ∈ [-0.5, 1]. Higher = better separation.
    """
    partition = community_louvain.best_partition(
        G, weight="weight", resolution=resolution
    )
    modularity = community_louvain.modularity(partition, G, weight="weight")

    n_communities = len(set(partition.values()))
    print(f"Communities detected: {n_communities}")
    print(f"Modularity score (Q): {modularity:.4f}")

    # Attach community labels to nodes
    nx.set_node_attributes(G, partition, "community")

    return partition, modularity


def compute_centrality_metrics(G: nx.Graph) -> Dict[str, Dict]:
    """
    Compute centrality metrics for all nodes.

    Returns
    -------
    dict with keys 'degree', 'betweenness', 'closeness', 'pagerank',
    each mapping node -> centrality value.
    """
    metrics = {
        "degree":      nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G, weight="weight"),
        "closeness":   nx.closeness_centrality(G),
        "pagerank":    nx.pagerank(G, weight="weight", alpha=0.85),
    }

    # Attach to graph as node attributes
    for metric_name, values in metrics.items():
        nx.set_node_attributes(G, values, f"{metric_name}_centrality")

    # Print top-5 by each metric
    for name, vals in metrics.items():
        top5 = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 by {name} centrality:")
        for node, score in top5:
            print(f"  {node}: {score:.4f}")

    return metrics
