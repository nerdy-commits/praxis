# =============================================================================
#  Healthcare-Grade Evaluation Metrics
# =============================================================================
"""
Evaluation metrics for DR grading with emphasis on clinical relevance.

Key metric: Quadratic Weighted Kappa (QWK)
    - Official APTOS 2019 competition metric
    - Penalizes misclassifications proportionally to their distance
      on the ordinal severity scale
    - Predicting 'No DR' when truth is 'Proliferative' is penalized
      much more than 'Moderate' vs 'Severe'
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK).

    QWK measures inter-rater agreement for ordinal scales.
    Range: -1 (complete disagreement) to 1 (perfect agreement).
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted class probabilities (N x num_classes) for AUC-ROC.

    Returns
    -------
    dict
        Dictionary of metric name → value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "quadratic_weighted_kappa": quadratic_weighted_kappa(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": float((y_true == y_pred).mean()),
    }

    # AUC-ROC (requires probability estimates)
    if y_prob is not None:
        try:
            metrics["auroc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
            metrics["auroc_weighted"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )
        except ValueError:
            metrics["auroc"] = None
            metrics["auroc_weighted"] = None

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    normalize: bool = True,
) -> None:
    """Generate and save a publication-ready confusion matrix heatmap."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor="#e0e0e0",
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title("DR Grading Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved: {save_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> None:
    """Generate one-vs-rest ROC curves for each DR grade."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        binary_true = (np.asarray(y_true) == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
        auc_val = roc_auc_score(binary_true, y_prob[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("One-vs-Rest ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curves saved: {save_path}")
