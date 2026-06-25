# =============================================================================
#  Network Analyzer — High-Level Analysis & Visualization
# =============================================================================
"""
Orchestrates network construction, analysis, and visualization.
Produces publication-ready network graphs and statistical summaries.
"""

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from .similarity import build_patient_network
from .community import detect_communities, compute_centrality_metrics


class NetworkAnalyzer:
    """
    End-to-end patient network analysis pipeline.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Patient clinical metadata.
    feature_columns : list of str
        Clinical features for similarity computation.
    threshold : float
        Edge creation threshold.
    dr_grades : np.ndarray, optional
        CNN-predicted DR severity grades.
    """

    DR_GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    DR_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

    def __init__(
        self,
        clinical_df: pd.DataFrame,
        feature_columns: list,
        threshold: float = 0.85,
        dr_grades: Optional[np.ndarray] = None,
    ):
        self.clinical_df = clinical_df
        self.feature_columns = feature_columns
        self.dr_grades = dr_grades

        # Build the network
        self.G = build_patient_network(
            clinical_df, feature_columns, threshold, dr_grades
        )

        # Run analyses
        self.partition, self.modularity = detect_communities(self.G)
        self.centrality = compute_centrality_metrics(self.G)

    def compute_dr_homophily(self) -> float:
        """
        Compute DR grade homophily — fraction of edges connecting
        same-grade patients. High homophily means the network
        naturally clusters patients by disease severity.
        """
        if self.dr_grades is None:
            return 0.0

        same_grade_edges = 0
        total_edges = self.G.number_of_edges()

        for u, v in self.G.edges():
            if self.G.nodes[u].get("dr_grade") == self.G.nodes[v].get("dr_grade"):
                same_grade_edges += 1

        homophily = same_grade_edges / max(total_edges, 1)
        print(f"DR grade homophily: {homophily:.4f}")
        return homophily

    def get_high_risk_clusters(self) -> Dict[int, Dict]:
        """
        Identify communities with high average DR grade.
        Returns summary statistics per community.
        """
        if self.dr_grades is None:
            return {}

        communities = {}
        for node, comm_id in self.partition.items():
            if comm_id not in communities:
                communities[comm_id] = {"grades": [], "nodes": []}
            grade = self.G.nodes[node].get("dr_grade", 0)
            communities[comm_id]["grades"].append(grade)
            communities[comm_id]["nodes"].append(node)

        summary = {}
        for comm_id, data in communities.items():
            grades = np.array(data["grades"])
            summary[comm_id] = {
                "size": len(data["nodes"]),
                "mean_dr_grade": float(grades.mean()),
                "max_dr_grade": int(grades.max()),
                "severe_fraction": float((grades >= 3).mean()),
            }

        # Sort by mean DR grade descending
        summary = dict(
            sorted(summary.items(), key=lambda x: x[1]["mean_dr_grade"], reverse=True)
        )
        return summary

    def visualize_network(
        self,
        save_path: str,
        color_by: str = "community",
        size_by: str = "degree",
        layout: str = "spring",
        figsize: tuple = (14, 10),
    ) -> None:
        """
        Generate a publication-ready network visualization.

        Parameters
        ----------
        save_path : str
            File path to save the figure.
        color_by : str
            'community' or 'dr_grade' — determines node coloring.
        size_by : str
            'degree', 'betweenness' — determines node sizing.
        layout : str
            'spring', 'kamada_kawai', or 'circular'.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Layout — tiered by graph size to avoid O(N²) RAM for large networks
        n_nodes = self.G.number_of_nodes()
        try:
            if layout == "spring" and n_nodes > 300:
                # For large graphs, start from a good random seed then do a
                # short Fruchterman-Reingold pass — cheap but looks far better
                # than circular_layout which puts all nodes on a ring.
                rng = np.random.default_rng(42)
                pos_init = {n: rng.uniform(-1, 1, 2) for n in self.G.nodes()}
                pos = nx.spring_layout(
                    self.G, pos=pos_init, fixed=None,
                    seed=42, k=1.2 / np.sqrt(n_nodes),
                    iterations=25
                )
            elif layout == "spring":
                pos = nx.spring_layout(self.G, seed=42, k=0.5, iterations=50)
            elif layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(self.G)
            elif layout == "circular":
                pos = nx.circular_layout(self.G)
            else:
                pos = nx.spring_layout(self.G, seed=42, k=0.5, iterations=30)
        except (MemoryError, Exception):
            # True last resort: random scatter is still better than a ring
            rng = np.random.default_rng(42)
            pos = {n: rng.uniform(-1, 1, 2) for n in self.G.nodes()}

        # Node sizes based on centrality — scale down for large graphs
        centrality_vals = self.centrality.get(size_by, self.centrality["degree"])
        sizes = np.array([centrality_vals.get(n, 0.01) for n in self.G.nodes()])
        # Smaller base + narrower range so nodes don't occlude each other at 500 nodes
        max_size = 300 if n_nodes > 200 else 500
        base_size = 20  if n_nodes > 200 else 60
        sizes = base_size + sizes / (sizes.max() + 1e-8) * max_size

        # Node colors
        if color_by == "dr_grade" and self.dr_grades is not None:
            colors = [
                self.DR_COLORS[self.G.nodes[n].get("dr_grade", 0)]
                for n in self.G.nodes()
            ]
        else:
            import matplotlib
            n_comm = len(set(self.partition.values()))
            cmap = matplotlib.colormaps["tab20"].resampled(n_comm)
            colors = [cmap(self.partition.get(n, 0)) for n in self.G.nodes()]

        # Draw
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")

        # Edges — slightly more visible than before
        nx.draw_networkx_edges(
            self.G, pos, ax=ax, alpha=0.10, edge_color="#5a5a8a", width=0.4
        )

        # Nodes — reduced alpha so overlapping nodes show through each other
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax, node_size=sizes, node_color=colors,
            edgecolors="none", linewidths=0.0, alpha=0.75
        )

        # Legend
        if color_by == "dr_grade":
            patches = [
                mpatches.Patch(color=c, label=n)
                for c, n in zip(self.DR_COLORS, self.DR_GRADE_NAMES)
            ]
            ax.legend(
                handles=patches, loc="upper left", fontsize=9,
                facecolor="#16213e", edgecolor="#4a4a6a", labelcolor="white"
            )

        ax.set_title(
            "Patient Comorbidity Risk Network",
            fontsize=16, fontweight="bold", color="white", pad=15
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Network visualization saved: {save_path}")

    def export_gexf(self, save_path: str) -> None:
        """Export network in GEXF format for Gephi visualization."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(self.G, str(save_path))
        print(f"GEXF exported: {save_path}")

    def export_pyvis(self, save_path: str = "outputs/network/patient_network.html") -> None:
        """
        Export an interactive HTML network using PyVis.

        Nodes are colour-coded by DR grade. Node size scales with
        betweenness centrality. Hovering shows patient stats.
        Opens directly in any browser — also embedded in Streamlit.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            print("  [WARN] pyvis not installed. Run: pip install pyvis")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        GRADE_COLORS = {
            0: "#2ecc71",   # Green  – No DR
            1: "#f1c40f",   # Yellow – Mild
            2: "#e67e22",   # Orange – Moderate
            3: "#e74c3c",   # Red    – Severe
            4: "#8e44ad",   # Purple – Proliferative
        }

        net = Network(
            height="750px", width="100%",
            bgcolor="#1a1a2e", font_color="white",
        )
        net.set_options("""{
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100}
          }
        }""")

        bc = self.centrality.get("betweenness", {})
        for node, data in self.G.nodes(data=True):
            grade  = data.get("dr_grade", 0)
            bc_val = bc.get(node, 0.01)
            size   = 10 + bc_val * 120
            color  = GRADE_COLORS.get(int(grade), "#95a5a6")
            title  = (
                f"ID: {node}<br>"
                f"DR Grade: {self.DR_GRADE_NAMES[int(grade)] if 0 <= int(grade) < 5 else grade}<br>"
                f"HbA1c: {data.get('hba1c', 'N/A')}<br>"
                f"Community: {self.partition.get(node, 'N/A')}<br>"
                f"Betweenness: {bc_val:.4f}"
            )
            net.add_node(node, label=str(node), color=color, size=size, title=title)

        # Prune dense network edges to prevent browser timeout/crash during layout rendering
        k = 4  # top-k strongest similarity connections per node
        strong_edges = set()
        for node in self.G.nodes():
            edges = self.G.edges(node, data=True)
            # Sort incident edges by weight descending and keep top k
            sorted_edges = sorted(edges, key=lambda x: x[2].get("weight", 0.0), reverse=True)[:k]
            for u, v, data in sorted_edges:
                # Store edge in sorted order to avoid duplicates
                edge = tuple(sorted([u, v]))
                strong_edges.add((edge[0], edge[1], data.get("weight", 0.5)))

        for u, v, w in strong_edges:
            net.add_edge(u, v, value=w)

        net.save_graph(str(save_path))
        print(f"Network visualization saved: {save_path}")

    def summary(self) -> Dict:
        """Return a summary dict of all network statistics."""
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "density": nx.density(self.G),
            "modularity": self.modularity,
            "n_communities": len(set(self.partition.values())),
            "avg_clustering": nx.average_clustering(self.G),
            "dr_homophily": self.compute_dr_homophily() if self.dr_grades is not None else None,
        }
