# =============================================================================
#  Visualization Utilities
# =============================================================================
"""
Plotting functions for training curves, class distributions,
and other diagnostic visualizations.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set global style
sns.set_theme(style="darkgrid", palette="deep")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
) -> None:
    """
    Plot training/validation loss and QWK curves side by side.

    Parameters
    ----------
    history : dict
        Training history with keys: train_loss, val_loss, train_qwk, val_qwk.
    save_path : str
        File path to save the figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # QWK curves
    ax2.plot(epochs, history["train_qwk"], "o-", label="Train QWK", markersize=3)
    ax2.plot(epochs, history["val_qwk"], "s-", label="Val QWK", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Quadratic Weighted Kappa")
    ax2.set_title("Training & Validation QWK", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved: {save_path}")


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: str,
    title: str = "DR Grade Distribution",
) -> None:
    """
    Plot class distribution as a styled bar chart.

    Parameters
    ----------
    class_counts : dict
        Mapping of class name → count.
    save_path : str
        File path to save the figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
    names = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, counts, color=colors[:len(names)], edgecolor="white", linewidth=0.8)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_xlabel("DR Severity Grade", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Class distribution saved: {save_path}")


def plot_centrality_distribution(
    centrality_values: Dict[str, float],
    metric_name: str,
    save_path: str,
) -> None:
    """Plot the distribution of a centrality metric across all nodes."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    values = list(centrality_values.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(values), color="#e74c3c", linestyle="--", label=f"Mean: {np.mean(values):.4f}")
    ax.set_xlabel(f"{metric_name.title()} Centrality", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"{metric_name.title()} Centrality Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
