"""
Regenerate the two static network PNGs with a proper spring layout.
Uses a stratified 500-node subsample (100 per DR grade) for speed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.network.similarity import build_patient_network
from src.network.community import detect_communities, compute_centrality_metrics

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/metadata/clinical_metadata.csv")
print("Columns:", list(df.columns))

# Stratified 500-node sample (100 per DR grade)
frames = []
for g in range(5):
    sub = df[df["dr_grade"] == g].sample(min(len(df[df["dr_grade"] == g]), 100), random_state=42)
    frames.append(sub)
sampled = pd.concat(frames, ignore_index=True)
print(f"Sampled {len(sampled)} patients across 5 DR grades")

FEATURE_COLS = ["hba1c", "diabetes_duration", "bmi", "bp_systolic", "age",
                "hypertension", "cardiovascular", "smoking"]
dr_grades = sampled["dr_grade"].values

# ── Graph ─────────────────────────────────────────────────────────────────────
print("Building graph …")
G = build_patient_network(sampled, FEATURE_COLS, threshold=0.85, dr_grades=dr_grades)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("Community detection …")
partition, modularity = detect_communities(G)
centrality = compute_centrality_metrics(G)
n_comm = len(set(partition.values()))
print(f"Communities: {n_comm}, Modularity Q={modularity:.4f}")

# ── Layout ────────────────────────────────────────────────────────────────────
print("Spring layout (500 nodes) …")
n = G.number_of_nodes()
pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(n), iterations=50)
print("Layout done.")

# ── Style constants ───────────────────────────────────────────────────────────
DR_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
DR_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
BG        = "#1a1a2e"

deg   = centrality.get("degree", {})
sizes = np.array([deg.get(nd, 0.01) for nd in G.nodes()])
sizes = 30 + sizes / (sizes.max() + 1e-8) * 280

# ── Community view ────────────────────────────────────────────────────────────
print("Saving community PNG …")
import matplotlib
cmap     = matplotlib.colormaps["tab20"].resampled(n_comm)
colors_c = [cmap(partition.get(nd, 0)) for nd in G.nodes()]

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.09, edge_color="#5a5a9a", width=0.5)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors_c,
                       edgecolors="none", alpha=0.82)
ax.set_title("Patient Comorbidity Risk Network",
             fontsize=16, fontweight="bold", color="white", pad=15)
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/network/patient_network_community.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("Community PNG saved.")

# ── DR grade view ─────────────────────────────────────────────────────────────
print("Saving DR grade PNG …")
colors_g = [DR_COLORS[int(G.nodes[nd].get("dr_grade", 0))] for nd in G.nodes()]

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.09, edge_color="#5a5a9a", width=0.5)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors_g,
                       edgecolors="none", alpha=0.82)
patches = [mpatches.Patch(color=c, label=name)
           for c, name in zip(DR_COLORS, DR_NAMES)]
ax.legend(handles=patches, loc="upper left", fontsize=9,
          facecolor="#16213e", edgecolor="#4a4a6a", labelcolor="white")
ax.set_title("Patient Comorbidity Risk Network",
             fontsize=16, fontweight="bold", color="white", pad=15)
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/network/patient_network_drgrade.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("DR grade PNG saved.")
print("All done.")
