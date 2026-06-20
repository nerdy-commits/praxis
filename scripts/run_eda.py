# =============================================================================
#  Phase 2 -- Exploratory Data Analysis
# =============================================================================
"""
Comprehensive EDA for the APTOS 2019 Diabetic Retinopathy dataset.

Generates:
    1. Class distribution bar chart (train/val/test)
    2. Sample retinal images per DR grade
    3. Ben Graham preprocessing before/after comparison
    4. Image dimension & channel statistics
    5. Clinical metadata distributions & correlations
    6. Summary statistics table

All outputs saved to outputs/figures/eda/

Usage:
    python scripts/run_eda.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import BenGrahamPreprocessor

# ── Config ───────────────────────────────────────────────────────────────────
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "aptos2019" / "train_1.csv"
VAL_CSV = PROJECT_ROOT / "data" / "raw" / "aptos2019" / "valid.csv"
TEST_CSV = PROJECT_ROOT / "data" / "raw" / "aptos2019" / "test.csv"
TRAIN_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "aptos2019" / "train_images" / "train_images"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "clinical_metadata.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

sns.set_theme(style="whitegrid", font_scale=1.1)


# =============================================================================
#  1. Class Distribution
# =============================================================================
def plot_class_distributions():
    """Plot class distribution across train/val/test splits."""
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DR Grade Distribution Across Splits", fontsize=16, fontweight="bold", y=1.02)

    for ax, df, title in zip(axes, [train, val, test], ["Train (n=2930)", "Validation (n=366)", "Test (n=366)"]):
        counts = df["diagnosis"].value_counts().sort_index()
        bars = ax.bar(CLASS_NAMES, [counts.get(i, 0) for i in range(5)],
                      color=CLASS_COLORS, edgecolor="white", linewidth=0.8)
        for bar, count in zip(bars, [counts.get(i, 0) for i in range(5)]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(count), ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("DR Grade")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[1/6] Class distribution plot saved")

    # Print imbalance ratios
    train_counts = train["diagnosis"].value_counts().sort_index()
    total = len(train)
    print("  Train class proportions:")
    for i, name in enumerate(CLASS_NAMES):
        c = train_counts.get(i, 0)
        print(f"    {name}: {c} ({c/total*100:.1f}%)")


# =============================================================================
#  2. Sample Images Per Grade
# =============================================================================
def plot_sample_images():
    """Show 3 random sample images for each DR grade."""
    train = pd.read_csv(TRAIN_CSV)
    np.random.seed(42)

    fig, axes = plt.subplots(5, 3, figsize=(12, 18))
    fig.suptitle("Sample Retinal Fundus Images by DR Grade", fontsize=16, fontweight="bold")

    for grade in range(5):
        grade_df = train[train["diagnosis"] == grade]
        samples = grade_df.sample(min(3, len(grade_df)), random_state=42)

        for j, (_, row) in enumerate(samples.iterrows()):
            img_path = TRAIN_IMG_DIR / f"{row['id_code']}.png"
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[grade, j].imshow(img)
            axes[grade, j].axis("off")
            if j == 0:
                axes[grade, j].set_ylabel(f"Grade {grade}\n{CLASS_NAMES[grade]}",
                                           fontsize=11, fontweight="bold", rotation=0,
                                           labelpad=80, va="center")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[2/6] Sample images grid saved")


# =============================================================================
#  3. Ben Graham Preprocessing Before/After
# =============================================================================
def plot_preprocessing_comparison():
    """Show before/after Ben Graham preprocessing for 4 images."""
    train = pd.read_csv(TRAIN_CSV)
    preprocessor = BenGrahamPreprocessor(image_size=224, sigma=10)

    # Pick one image per grade (0-3)
    samples = []
    for grade in range(4):
        grade_df = train[train["diagnosis"] == grade]
        samples.append(grade_df.iloc[0])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Ben Graham Preprocessing: Before vs After", fontsize=16, fontweight="bold")

    for i, row in enumerate(samples):
        img_path = TRAIN_IMG_DIR / f"{row['id_code']}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_processed = preprocessor(img_rgb)

        axes[0, i].imshow(img_resized)
        axes[0, i].set_title(f"Grade {int(row['diagnosis'])}: Original", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(img_processed)
        axes[1, i].set_title(f"Grade {int(row['diagnosis'])}: Processed", fontsize=10)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "preprocessing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[3/6] Preprocessing comparison saved")


# =============================================================================
#  4. Image Statistics
# =============================================================================
def compute_image_statistics():
    """Compute image dimension and pixel intensity statistics."""
    train = pd.read_csv(TRAIN_CSV)

    # Sample 200 images for speed
    sample = train.sample(min(200, len(train)), random_state=42)
    heights, widths, means, stds = [], [], [], []

    for _, row in sample.iterrows():
        img_path = TRAIN_IMG_DIR / f"{row['id_code']}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        heights.append(h)
        widths.append(w)
        img_float = img.astype(np.float32) / 255.0
        means.append(img_float.mean(axis=(0, 1)))
        stds.append(img_float.std(axis=(0, 1)))

    means = np.array(means)
    stds = np.array(stds)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Image Statistics (200 sample images)", fontsize=14, fontweight="bold")

    # Dimensions
    axes[0].scatter(widths, heights, alpha=0.4, s=15, color="#3498db")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].set_title("Image Dimensions")
    axes[0].grid(alpha=0.3)

    # Channel means
    channels = ["Blue", "Green", "Red"]
    ch_colors = ["#3498db", "#2ecc71", "#e74c3c"]
    for ch in range(3):
        axes[1].hist(means[:, ch], bins=25, alpha=0.5, color=ch_colors[ch], label=channels[ch])
    axes[1].set_xlabel("Mean Pixel Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Channel Mean Distribution")
    axes[1].legend()

    # Channel stds
    for ch in range(3):
        axes[2].hist(stds[:, ch], bins=25, alpha=0.5, color=ch_colors[ch], label=channels[ch])
    axes[2].set_xlabel("Std Pixel Value")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Channel Std Distribution")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "image_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[4/6] Image statistics saved")

    print(f"  Dimensions: median {int(np.median(widths))}x{int(np.median(heights))}")
    print(f"  Channel means (BGR): {means.mean(axis=0).round(3)}")
    print(f"  Channel stds  (BGR): {stds.mean(axis=0).round(3)}")


# =============================================================================
#  5. Clinical Metadata Analysis
# =============================================================================
def plot_clinical_metadata():
    """Analyze and visualize synthetic clinical metadata."""
    meta = pd.read_csv(METADATA_CSV)
    continuous_features = ["age", "hba1c", "bmi", "bp_systolic", "bp_diastolic",
                           "diabetes_duration", "cholesterol"]

    # 5a. Feature distributions by DR grade
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Clinical Feature Distributions by DR Grade", fontsize=16, fontweight="bold")
    axes_flat = axes.flatten()

    for i, feat in enumerate(continuous_features):
        ax = axes_flat[i]
        for grade in range(5):
            data = meta[meta["dr_grade"] == grade][feat]
            ax.hist(data, bins=20, alpha=0.4, color=CLASS_COLORS[grade],
                    label=CLASS_NAMES[grade], density=True)
        ax.set_title(feat.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")

    # Legend in last subplot
    axes_flat[7].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in CLASS_COLORS]
    axes_flat[7].legend(handles, CLASS_NAMES, loc="center", fontsize=12, title="DR Grade")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "clinical_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5b. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_cols = continuous_features + ["dr_grade"]
    corr = meta[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5,
                xticklabels=[c.replace("_", " ").title() for c in corr_cols],
                yticklabels=[c.replace("_", " ").title() for c in corr_cols])
    ax.set_title("Clinical Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "clinical_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("[5/6] Clinical metadata analysis saved")

    # Print summary stats
    print("  Clinical metadata summary:")
    print(meta[continuous_features].describe().round(2).to_string())


# =============================================================================
#  6. Summary Report
# =============================================================================
def generate_summary():
    """Print a comprehensive EDA summary."""
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)
    meta = pd.read_csv(METADATA_CSV)

    print("\n" + "=" * 60)
    print("  EDA Summary Report")
    print("=" * 60)
    print(f"\n  Dataset: APTOS 2019 Blindness Detection (Kaggle mirror)")
    print(f"  Total patients: {len(train) + len(val) + len(test)}")
    print(f"  Splits: Train={len(train)} | Val={len(val)} | Test={len(test)}")
    print(f"  Classes: 5 (No DR, Mild, Moderate, Severe, Proliferative)")
    print(f"  Clinical features: {len(meta.columns) - 2} (+ patient_id + dr_grade)")
    print(f"\n  Class imbalance ratio (max/min): "
          f"{train['diagnosis'].value_counts().max() / train['diagnosis'].value_counts().min():.1f}x")
    print(f"\n  All EDA figures saved to: {OUT_DIR}")
    print("[6/6] Summary complete")


# =============================================================================
#  Main
# =============================================================================
if __name__ == "__main__":
    print("Praxis -- Phase 2: Exploratory Data Analysis\n")

    plot_class_distributions()
    plot_sample_images()
    plot_preprocessing_comparison()
    compute_image_statistics()
    plot_clinical_metadata()
    generate_summary()
