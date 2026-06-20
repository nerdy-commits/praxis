# =============================================================================
#  Patient Similarity Network Construction
# =============================================================================
"""
Builds a patient similarity graph from clinical feature vectors.

Each patient is a node. Edges connect patients whose cosine similarity
exceeds a threshold, weighted by that similarity score.

This produces a graph where clinically similar patients cluster together,
enabling community detection to surface high-risk subgroups.
"""

from typing import Optional, List

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def build_patient_network(
    clinical_df: pd.DataFrame,
    feature_columns: List[str],
    threshold: float = 0.85,
    dr_grades: Optional[np.ndarray] = None,
    patient_ids: Optional[List[str]] = None,
) -> nx.Graph:
    """
    Construct a patient similarity network from clinical metadata.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        DataFrame with patient clinical features.
    feature_columns : list of str
        Column names to use for similarity computation.
    threshold : float
        Minimum cosine similarity to create an edge (0–1).
    dr_grades : np.ndarray, optional
        CNN-predicted DR grades to attach as node attributes.
    patient_ids : list of str, optional
        Patient identifiers. If None, uses DataFrame index.

    Returns
    -------
    nx.Graph
        Undirected weighted graph with patient nodes and similarity edges.
    """
    # Extract and normalize features
    features = clinical_df[feature_columns].values.astype(np.float64)
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)

    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(features_norm)

    # Build graph
    n_patients = len(clinical_df)
    ids = patient_ids if patient_ids else [f"P{i:04d}" for i in range(n_patients)]

    G = nx.Graph()

    # Add nodes with attributes
    for i, pid in enumerate(ids):
        attrs = {col: float(clinical_df.iloc[i][col]) for col in feature_columns}
        if dr_grades is not None:
            attrs["dr_grade"] = int(dr_grades[i])
        G.add_node(pid, **attrs)

    # Add edges where similarity > threshold
    edge_count = 0
    for i in range(n_patients):
        for j in range(i + 1, n_patients):
            if sim_matrix[i, j] > threshold:
                G.add_edge(ids[i], ids[j], weight=float(sim_matrix[i, j]))
                edge_count += 1

    print(f"Network built: {G.number_of_nodes()} nodes, {edge_count} edges")
    print(f"Edge density: {nx.density(G):.4f}")

    return G
