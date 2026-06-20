"""
Generate all 6 Jupyter notebooks for the Praxis project.
Run from the project root:  python scripts/create_notebooks.py
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def nb(cells):
    """Create a minimal nbformat v4 notebook dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": "md"}


def code(source, outputs=None):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": outputs or [],
        "execution_count": None,
        "id": "code",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 01 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
nb01 = nb([
    md("# 01 — Exploratory Data Analysis\n\n"
       "This notebook performs a full EDA on the APTOS 2019 dataset and the synthetic "
       "clinical metadata, covering:\n\n"
       "1. Dataset overview & class distribution\n"
       "2. Sample images per DR grade\n"
       "3. Ben Graham preprocessing before/after\n"
       "4. Image statistics (dimensions, pixel distributions)\n"
       "5. Clinical metadata distributions & correlations\n\n"
       "> **Prerequisites**: Run `python scripts/download_data.py` first."),

    code("import sys\nfrom pathlib import Path\n"
         "sys.path.insert(0, str(Path('..').resolve()))\n\n"
         "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
         "import seaborn as sns\nimport cv2\n\n"
         "sns.set_theme(style='darkgrid', palette='muted')\n"
         "%matplotlib inline\nprint('Imports OK')"),

    md("## 1. Dataset Overview"),

    code("TRAIN_CSV = Path('../data/raw/aptos2019/train_split.csv')\n"
         "VAL_CSV   = Path('../data/raw/aptos2019/val_split.csv')\n"
         "TEST_CSV  = Path('../data/raw/aptos2019/test_split.csv')\n"
         "TRAIN_IMG = Path('../data/raw/aptos2019/train_images/train_images')\n\n"
         "CLASS_NAMES  = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']\n"
         "CLASS_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']\n\n"
         "train = pd.read_csv(TRAIN_CSV)\nval = pd.read_csv(VAL_CSV)\ntest = pd.read_csv(TEST_CSV)\n\n"
         "print(f'Train: {len(train)} | Val: {len(val)} | Test: {len(test)}')\n"
         "train.head()"),

    md("## 2. Class Distribution"),

    code("fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
         "fig.suptitle('DR Grade Distribution Across Splits', fontsize=16, fontweight='bold')\n\n"
         "for ax, df, title in zip(axes, [train, val, test],\n"
         "                          ['Train', 'Validation', 'Test']):\n"
         "    counts = df['diagnosis'].value_counts().sort_index()\n"
         "    bars = ax.bar(CLASS_NAMES, [counts.get(i, 0) for i in range(5)],\n"
         "                  color=CLASS_COLORS, edgecolor='white', linewidth=0.8)\n"
         "    for bar, c in zip(bars, [counts.get(i, 0) for i in range(5)]):\n"
         "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,\n"
         "                str(c), ha='center', va='bottom', fontweight='bold')\n"
         "    ax.set_title(title, fontweight='bold')\n"
         "    ax.tick_params(axis='x', rotation=30)\n"
         "plt.tight_layout()\n"
         "plt.savefig('../outputs/figures/eda_class_distribution.png', dpi=150, bbox_inches='tight')\n"
         "plt.show()"),

    code("# Imbalance analysis\ncounts = train['diagnosis'].value_counts().sort_index()\n"
         "ratio = counts.max() / counts.min()\nprint(f'Imbalance ratio (max/min): {ratio:.1f}x')\n"
         "print(counts.to_string())"),

    md("## 3. Sample Images Per Grade\n\n"
       "Visualise 3 representative fundus images for each DR severity class."),

    code("np.random.seed(42)\n"
         "fig, axes = plt.subplots(5, 3, figsize=(12, 18))\n"
         "fig.suptitle('Sample Retinal Fundus Images by DR Grade', fontsize=16, fontweight='bold')\n\n"
         "for grade in range(5):\n"
         "    grade_df = train[train['diagnosis'] == grade]\n"
         "    samples = grade_df.sample(min(3, len(grade_df)), random_state=42)\n"
         "    for j, (_, row) in enumerate(samples.iterrows()):\n"
         "        img_path = TRAIN_IMG / f\"{row['id_code']}.png\"\n"
         "        img = cv2.imread(str(img_path))\n"
         "        if img is not None:\n"
         "            axes[grade, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
         "        axes[grade, j].axis('off')\n"
         "        if j == 0:\n"
         "            axes[grade, j].set_ylabel(f'Grade {grade}\\n{CLASS_NAMES[grade]}',\n"
         "                                       fontsize=11, fontweight='bold',\n"
         "                                       rotation=0, labelpad=80, va='center')\n"
         "plt.tight_layout()\n"
         "plt.savefig('../outputs/figures/eda_sample_images.png', dpi=150, bbox_inches='tight')\n"
         "plt.show()"),

    md("## 4. Ben Graham Preprocessing: Before vs After\n\n"
       "Ben Graham's method enhances retinal vessel contrast by subtracting a blurred "
       "version and adding a neutral grey offset."),

    code("from src.data.preprocessing import BenGrahamPreprocessor\n\n"
         "preprocessor = BenGrahamPreprocessor(image_size=224, sigma=10)\n\n"
         "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n"
         "fig.suptitle('Ben Graham Preprocessing: Before vs After', fontsize=16, fontweight='bold')\n\n"
         "for i, grade in enumerate(range(4)):\n"
         "    row = train[train['diagnosis'] == grade].iloc[0]\n"
         "    img_path = TRAIN_IMG / f\"{row['id_code']}.png\"\n"
         "    img = cv2.imread(str(img_path))\n"
         "    if img is None:\n"
         "        continue\n"
         "    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n"
         "    img_proc = preprocessor(img_rgb)\n\n"
         "    axes[0, i].imshow(cv2.resize(img_rgb, (224, 224)))\n"
         "    axes[0, i].set_title(f'Grade {grade}: Original', fontsize=10)\n"
         "    axes[0, i].axis('off')\n"
         "    axes[1, i].imshow(img_proc)\n"
         "    axes[1, i].set_title(f'Grade {grade}: Processed', fontsize=10)\n"
         "    axes[1, i].axis('off')\n\n"
         "plt.tight_layout()\n"
         "plt.savefig('../outputs/figures/eda_preprocessing.png', dpi=150, bbox_inches='tight')\n"
         "plt.show()"),

    md("## 5. Clinical Metadata Analysis"),

    code("meta = pd.read_csv('../data/metadata/clinical_metadata.csv')\n"
         "print(f'Metadata shape: {meta.shape}')\n"
         "meta.head()"),

    code("# Feature distributions by DR grade\n"
         "continuous = ['age', 'hba1c', 'bmi', 'bp_systolic', 'diabetes_duration', 'cholesterol']\n"
         "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n"
         "fig.suptitle('Clinical Features by DR Grade', fontsize=16, fontweight='bold')\n\n"
         "for ax, feat in zip(axes.flat, continuous):\n"
         "    for grade in range(5):\n"
         "        data = meta[meta['dr_grade'] == grade][feat]\n"
         "        ax.hist(data, bins=25, alpha=0.45, color=CLASS_COLORS[grade],\n"
         "                label=CLASS_NAMES[grade], density=True)\n"
         "    ax.set_title(feat.replace('_', ' ').title(), fontweight='bold')\n"
         "    ax.set_xlabel(feat)\n"
         "    ax.set_ylabel('Density')\n\n"
         "axes.flat[0].legend(fontsize=9)\n"
         "plt.tight_layout()\n"
         "plt.savefig('../outputs/figures/eda_clinical_distributions.png', dpi=150, bbox_inches='tight')\n"
         "plt.show()"),

    code("# Correlation heatmap\nfig, ax = plt.subplots(figsize=(10, 8))\n"
         "corr = meta[continuous + ['dr_grade']].corr()\n"
         "mask = np.triu(np.ones_like(corr, dtype=bool))\n"
         "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',\n"
         "            center=0, ax=ax, linewidths=0.5)\n"
         "ax.set_title('Clinical Feature Correlation Matrix', fontsize=14, fontweight='bold')\n"
         "plt.tight_layout()\n"
         "plt.savefig('../outputs/figures/eda_correlation.png', dpi=150, bbox_inches='tight')\n"
         "plt.show()"),

    md("## Key Findings\n\n"
       "- **Class Imbalance**: Grade 0 (No DR) is ~49% of all samples — requires class weighting\n"
       "- **Clinical Correlations**: HbA1c and diabetes duration are most correlated with DR grade\n"
       "- **Image Variability**: Raw images have varying resolutions — Ben Graham normalizes this\n"
       "- **Preprocessing Effect**: The Ben Graham filter reveals retinal vessels more clearly"),
])

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 02 — Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════
nb02 = nb([
    md("# 02 — Preprocessing Pipeline\n\n"
       "This notebook validates and benchmarks the full preprocessing pipeline:\n\n"
       "1. Ben Graham preprocessing — parameter sweep (sigma: 5, 10, 20)\n"
       "2. Augmentation pipeline — visual verification\n"
       "3. Batch preprocessing — time & quality check\n"
       "4. DataLoader construction — verify shapes & normalization"),

    code("import sys\nfrom pathlib import Path\n"
         "sys.path.insert(0, str(Path('..').resolve()))\n\n"
         "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport cv2\nimport torch\n\n"
         "from src.data.preprocessing import BenGrahamPreprocessor\n"
         "from src.data.augmentation import get_train_transforms, get_val_transforms\n"
         "from src.data.dataset import APTOSDataset\n"
         "from src.utils.config import load_config\n\n"
         "cfg = load_config('../configs/default.yaml')\nprint('Config loaded')"),

    md("## 1. Ben Graham Sigma Sweep"),

    code("TRAIN_IMG = Path('../data/raw/aptos2019/train_images/train_images')\n"
         "TRAIN_CSV = Path('../data/raw/aptos2019/train_split.csv')\n"
         "train = pd.read_csv(TRAIN_CSV)\n\n"
         "sample_path = TRAIN_IMG / f\"{train.iloc[100]['id_code']}.png\"\n"
         "img = cv2.cvtColor(cv2.imread(str(sample_path)), cv2.COLOR_BGR2RGB)\n\n"
         "sigmas = [5, 10, 20]\n"
         "fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(18, 4))\n"
         "axes[0].imshow(cv2.resize(img, (224, 224)))\n"
         "axes[0].set_title('Original (resized)', fontweight='bold')\n"
         "axes[0].axis('off')\n\n"
         "for ax, sigma in zip(axes[1:], sigmas):\n"
         "    preprocessor = BenGrahamPreprocessor(224, sigma=sigma)\n"
         "    ax.imshow(preprocessor(img))\n"
         "    ax.set_title(f'sigma={sigma}', fontweight='bold')\n"
         "    ax.axis('off')\n\n"
         "fig.suptitle('Ben Graham σ Sweep', fontsize=14, fontweight='bold')\n"
         "plt.tight_layout()\n"
         "plt.savefig('../outputs/figures/preprocessing_sigma_sweep.png', dpi=150, bbox_inches='tight')\n"
         "plt.show()\n"
         "print('sigma=10 is the literature default (Ben Graham, 2015)')"),

    md("## 2. Augmentation Pipeline Verification\n\n"
       "Visual confirmation that augmentations are reasonable for retinal images."),

    code("transform = get_train_transforms(224)\n\n"
         "fig, axes = plt.subplots(2, 5, figsize=(18, 7))\n"
         "fig.suptitle('Training Augmentation Samples', fontsize=14, fontweight='bold')\n\n"
         "for i, ax in enumerate(axes.flat):\n"
         "    augmented = transform(image=img)\n"
         "    img_t = augmented['image']\n"
         "    # Denormalise for display\n"
         "    mean = np.array([0.485, 0.456, 0.406])\n"
         "    std  = np.array([0.229, 0.224, 0.225])\n"
         "    img_display = img_t.permute(1,2,0).numpy() * std + mean\n"
         "    img_display = np.clip(img_display, 0, 1)\n"
         "    ax.imshow(img_display)\n"
         "    ax.axis('off')\n"
         "    ax.set_title(f'Aug {i+1}', fontsize=9)\n\n"
         "plt.tight_layout()\n"
         "plt.savefig('../outputs/figures/augmentation_samples.png', dpi=150, bbox_inches='tight')\n"
         "plt.show()"),

    md("## 3. DataLoader Shape Verification"),

    code("from src.data.dataset import create_data_loaders\n\n"
         "train_loader, val_loader, test_loader = create_data_loaders(\n"
         "    train_csv='../data/raw/aptos2019/train_split.csv',\n"
         "    val_csv='../data/raw/aptos2019/val_split.csv',\n"
         "    test_csv='../data/raw/aptos2019/test_split.csv',\n"
         "    train_image_dir='../data/raw/aptos2019/train_images/train_images',\n"
         "    val_image_dir='../data/raw/aptos2019/val_images/val_images',\n"
         "    test_image_dir='../data/raw/aptos2019/test_images/test_images',\n"
         "    train_transform=get_train_transforms(224),\n"
         "    val_transform=get_val_transforms(224),\n"
         "    image_size=224,\n"
         "    batch_size=8,\n"
         ")\n\n"
         "images, labels = next(iter(train_loader))\n"
         "print(f'Image batch shape: {images.shape}')\n"
         "print(f'Label batch shape: {labels.shape}')\n"
         "print(f'Pixel range: [{images.min():.3f}, {images.max():.3f}]')\n"
         "print(f'Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}')"),
])

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 03 — Model Training
# ═══════════════════════════════════════════════════════════════════════════════
nb03 = nb([
    md("# 03 — Model Training\n\n"
       "This notebook trains and compares three models:\n"
       "1. **ResNet-50** (primary) — two-phase fine-tuning\n"
       "2. **VGG-16** — transfer learning baseline\n"
       "3. **ResNet-50 from scratch** — no-pretrain baseline\n\n"
       "Evaluation metric: **Quadratic Weighted Kappa (QWK)**"),

    code("import sys\nfrom pathlib import Path\n"
         "sys.path.insert(0, str(Path('..').resolve()))\n\n"
         "import numpy as np\nimport torch\nimport matplotlib.pyplot as plt\n\n"
         "from src.utils.config import load_config, get_device\n"
         "from src.data import get_train_transforms, get_val_transforms\n"
         "from src.data.dataset import create_data_loaders\n"
         "from src.models import build_resnet50, build_vgg16, Trainer\n\n"
         "cfg    = load_config('../configs/default.yaml')\n"
         "device = get_device()\n"
         "print(f'Using device: {device}')"),

    md("## 1. Data Loading"),

    code("train_loader, val_loader, test_loader = create_data_loaders(\n"
         "    train_csv=cfg['paths']['train_csv'],\n"
         "    val_csv=cfg['paths']['val_csv'],\n"
         "    test_csv=cfg['paths']['test_csv'],\n"
         "    train_image_dir=cfg['paths']['train_images'],\n"
         "    val_image_dir=cfg['paths']['val_images'],\n"
         "    test_image_dir=cfg['paths']['test_images'],\n"
         "    train_transform=get_train_transforms(cfg['data']['image_size']),\n"
         "    val_transform=get_val_transforms(cfg['data']['image_size']),\n"
         "    image_size=cfg['data']['image_size'],\n"
         "    batch_size=cfg['training']['batch_size'],\n"
         ")\n"
         "print(f'Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}')"),

    md("## 2. ResNet-50 Training (Primary Model)"),

    code("model_r50 = build_resnet50(\n"
         "    num_classes=5, fc_hidden=512, dropout=0.3,\n"
         "    pretrained=True, unfreeze_blocks=2,\n"
         ")\n"
         "# Print parameter count\n"
         "total   = sum(p.numel() for p in model_r50.parameters())\n"
         "trainable = sum(p.numel() for p in model_r50.parameters() if p.requires_grad)\n"
         "print(f'Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)')"),

    code("trainer_r50 = Trainer(model_r50, device, cfg, save_dir='../outputs/models')\n"
         "history_r50 = trainer_r50.train(train_loader, val_loader)\n"
         "print('ResNet-50 training complete!')"),

    md("## 3. Training Curves"),

    code("from src.utils.visualization import plot_training_curves\n\n"
         "plot_training_curves(history_r50, '../outputs/figures/training_curves_r50.png')\n\n"
         "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
         "axes[0].plot(history_r50['train_loss'], label='Train Loss')\n"
         "axes[0].plot(history_r50['val_loss'], label='Val Loss')\n"
         "axes[0].set_title('Loss', fontweight='bold')\n"
         "axes[0].legend()\n\n"
         "axes[1].plot(history_r50['train_qwk'], label='Train QWK')\n"
         "axes[1].plot(history_r50['val_qwk'], label='Val QWK')\n"
         "axes[1].set_title('Quadratic Weighted Kappa', fontweight='bold')\n"
         "axes[1].legend()\n\n"
         "plt.tight_layout()\nplt.savefig('../outputs/figures/training_curves.png', dpi=150, bbox_inches='tight')\n"
         "plt.show()\n"
         "print(f'Best Val QWK: {max(history_r50[\"val_qwk\"]):.4f}')"),

    md("## 4. VGG-16 Baseline Comparison\n\n"
       "Train VGG-16 on the same data for comparison."),

    code("from src.models import build_vgg16\n\n"
         "model_vgg = build_vgg16(num_classes=5, fc_hidden=512, dropout=0.3, pretrained=True)\n"
         "trainer_vgg = Trainer(model_vgg, device, cfg, save_dir='../outputs/models')\n"
         "# Reduce epochs for quick baseline\n"
         "cfg_quick = dict(cfg)\n"
         "cfg_quick['training'] = dict(cfg['training'])\n"
         "cfg_quick['training']['epochs'] = 15\n"
         "history_vgg = trainer_vgg.train(train_loader, val_loader)\n"
         "print('VGG-16 training complete!')"),
])

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 04 — Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════════
nb04 = nb([
    md("# 04 — Grad-CAM Explainability\n\n"
       "This notebook generates and analyses Grad-CAM heatmaps for the trained ResNet-50.\n\n"
       "- Single image heatmap generation\n"
       "- Batch heatmaps across all DR grades\n"
       "- Failure case analysis (mis-graded images)\n"
       "- Clinical interpretation of activation regions"),

    code("import sys\nfrom pathlib import Path\n"
         "sys.path.insert(0, str(Path('..').resolve()))\n\n"
         "import numpy as np\nimport torch\nimport matplotlib.pyplot as plt\nimport cv2\n\n"
         "from src.utils.config import load_config, get_device\n"
         "from src.models import build_resnet50\n"
         "from src.data import get_val_transforms\n"
         "from src.data.dataset import create_data_loaders\n"
         "from src.explainability.gradcam import GradCAMVisualizer\n\n"
         "cfg    = load_config('../configs/default.yaml')\n"
         "device = get_device()\n\n"
         "# Load trained model\n"
         "model = build_resnet50(num_classes=5, fc_hidden=512, dropout=0.3,\n"
         "                        pretrained=False).to(device)\n"
         "ckpt = torch.load('../outputs/models/best_model.pth', map_location=device)\n"
         "model.load_state_dict(ckpt['model_state_dict'])\n"
         "model.eval()\n"
         "print(f'Model loaded — best epoch: {ckpt[\"epoch\"]} | Val QWK: {ckpt[\"val_qwk\"]:.4f}')"),

    md("## 1. Single Image Grad-CAM"),

    code("TRAIN_IMG = Path('../data/raw/aptos2019/train_images/train_images')\n"
         "import pandas as pd\n"
         "train = pd.read_csv('../data/raw/aptos2019/train_split.csv')\n\n"
         "# Pick a Grade 3 (Severe) example\n"
         "row = train[train['diagnosis'] == 3].iloc[0]\n"
         "img_path = TRAIN_IMG / f\"{row['id_code']}.png\"\n"
         "img_bgr = cv2.imread(str(img_path))\n"
         "img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n\n"
         "transform = get_val_transforms(224)\n"
         "augmented = transform(image=img_rgb)\n"
         "tensor = augmented['image'].unsqueeze(0).to(device)\n\n"
         "cam = GradCAMVisualizer(model, device)\n"
         "heatmap = cam.explainer.generate_heatmap(tensor)\n"
         "heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))\n"
         "cam_overlay = cam.explainer.overlay_heatmap(img_rgb, heatmap_resized)\n\n"
         "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
         "axes[0].imshow(cv2.resize(img_rgb, (224, 224)))\n"
         "axes[0].set_title('Original Image', fontweight='bold')\n"
         "axes[0].axis('off')\n"
         "axes[1].imshow(heatmap, cmap='jet')\n"
         "axes[1].set_title('Grad-CAM Heatmap', fontweight='bold')\n"
         "axes[1].axis('off')\n"
         "axes[2].imshow(cam_overlay)\n"
         "axes[2].set_title('Overlay (Grade 3 — Severe DR)', fontweight='bold')\n"
         "axes[2].axis('off')\n"
         "plt.tight_layout()\nplt.savefig('../outputs/figures/gradcam_single.png', dpi=150, bbox_inches='tight')\n"
         "plt.show()"),

    md("## 2. Batch Grad-CAM Across All Grades"),

    code("_, _, test_loader = create_data_loaders(\n"
         "    train_csv=cfg['paths']['train_csv'],\n"
         "    val_csv=cfg['paths']['val_csv'],\n"
         "    test_csv=cfg['paths']['test_csv'],\n"
         "    train_image_dir=cfg['paths']['train_images'],\n"
         "    val_image_dir=cfg['paths']['val_images'],\n"
         "    test_image_dir=cfg['paths']['test_images'],\n"
         "    train_transform=get_val_transforms(224),\n"
         "    val_transform=get_val_transforms(224),\n"
         "    image_size=224, batch_size=32,\n"
         ")\n\n"
         "cam.batch_generate(\n"
         "    test_loader,\n"
         "    n_samples=cfg['explainability']['num_samples'],\n"
         "    output_dir='../outputs/gradcam/',\n"
         ")\n"
         "print('Batch Grad-CAM generation complete!')"),

    md("## 3. Display Saved Heatmaps"),

    code("gradcam_dir = Path('../outputs/gradcam')\n"
         "heatmap_files = sorted(gradcam_dir.glob('*.png'))[:10]\n\n"
         "n = len(heatmap_files)\n"
         "if n > 0:\n"
         "    fig, axes = plt.subplots(2, 5, figsize=(20, 8))\n"
         "    for ax, fp in zip(axes.flat, heatmap_files):\n"
         "        ax.imshow(cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB))\n"
         "        ax.set_title(fp.stem[:25], fontsize=7)\n"
         "        ax.axis('off')\n"
         "    plt.suptitle('Grad-CAM Heatmaps', fontsize=14, fontweight='bold')\n"
         "    plt.tight_layout()\n"
         "    plt.show()\nelse:\n"
         "    print('No heatmaps yet — run Step 2 above')"),

    md("## Clinical Interpretation\n\n"
       "| DR Grade | Expected Activated Regions |\n"
       "|---|---|\n"
       "| 0 — No DR | Background / disc area (model uncertain) |\n"
       "| 1 — Mild | Near optic disc: microaneurysms |\n"
       "| 2 — Moderate | Perifoveal hemorrhages, hard exudates |\n"
       "| 3 — Severe | Multiple quadrants: haemorrhages, IRMA |\n"
       "| 4 — Proliferative | Neovascularisation, vitreous haemorrhage |\n\n"
       "> 🔴 Red/yellow regions = high Grad-CAM activation (important for prediction)  \n"
       "> The XAI layer enables **clinical trust** — clinicians can verify that the model "
       "attends to genuine lesion regions rather than artefacts."),
])

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 05 — Network Analysis
# ═══════════════════════════════════════════════════════════════════════════════
nb05 = nb([
    md("# 05 — Patient Similarity Network Analysis\n\n"
       "This notebook builds and analyses the patient comorbidity risk network:\n\n"
       "1. Feature vector construction from clinical metadata\n"
       "2. Patient similarity graph construction (cosine similarity)\n"
       "3. Louvain community detection\n"
       "4. Centrality analysis (degree, betweenness, PageRank)\n"
       "5. DR grade homophily measurement\n"
       "6. High-risk cluster identification\n"
       "7. Gephi export for interactive exploration"),

    code("import sys\nfrom pathlib import Path\n"
         "sys.path.insert(0, str(Path('..').resolve()))\n\n"
         "import numpy as np\nimport pandas as pd\nimport networkx as nx\nimport matplotlib.pyplot as plt\n\n"
         "from src.network.similarity import build_patient_network\n"
         "from src.network.community import detect_communities, compute_centrality_metrics\n"
         "from src.network.analysis import NetworkAnalyzer\n"
         "from src.utils.config import load_config\n\n"
         "cfg = load_config('../configs/default.yaml')\n"
         "clinical_df = pd.read_csv('../data/metadata/clinical_metadata.csv')\n"
         "print(f'Loaded {len(clinical_df)} patients')\n"
         "clinical_df.head()"),

    md("## 1. Feature Vector Construction"),

    code("feature_cols = [c for c in cfg['network']['clinical_features']\n"
         "                if c in clinical_df.columns]\n"
         "print(f'Using {len(feature_cols)} features: {feature_cols}')\n"
         "clinical_df[feature_cols].describe().round(2)"),

    md("## 2. Build Patient Similarity Graph"),

    code("dr_grades = clinical_df['dr_grade'].values if 'dr_grade' in clinical_df.columns else None\n\n"
         "G = build_patient_network(\n"
         "    clinical_df=clinical_df,\n"
         "    feature_columns=feature_cols,\n"
         "    threshold=cfg['network']['edge_threshold'],\n"
         "    dr_grades=dr_grades,\n"
         ")\n\n"
         "print(f'\\nGraph statistics:')\n"
         "print(f'  Nodes: {G.number_of_nodes()}')\n"
         "print(f'  Edges: {G.number_of_edges()}')\n"
         "print(f'  Density: {nx.density(G):.4f}')\n"
         "print(f'  Avg clustering: {nx.average_clustering(G):.4f}')"),

    md("## 3. Community Detection (Louvain)"),

    code("partition, modularity = detect_communities(G)\n"
         "n_communities = len(set(partition.values()))\n"
         "print(f'Communities detected: {n_communities}')\n"
         "print(f'Modularity score (Q): {modularity:.4f}')\n"
         "print('  Q > 0.3 indicates meaningful community structure')\n\n"
         "# Community size distribution\n"
         "from collections import Counter\n"
         "comm_sizes = Counter(partition.values())\n"
         "print(f'\\nCommunity sizes: {sorted(comm_sizes.values(), reverse=True)[:10]}')"),

    md("## 4. Centrality Analysis"),

    code("centrality = compute_centrality_metrics(G)\n\n"
         "# Top 10 most connected patients\n"
         "top_degree = sorted(centrality['degree'].items(), key=lambda x: -x[1])[:10]\n"
         "print('Top 10 by Degree Centrality:')\n"
         "for pid, dc in top_degree:\n"
         "    grade = G.nodes[pid].get('dr_grade', '?')\n"
         "    print(f'  {pid}: degree={dc:.4f}  DR_grade={grade}')\n\n"
         "# Top 10 bridge patients\n"
         "top_between = sorted(centrality['betweenness'].items(), key=lambda x: -x[1])[:10]\n"
         "print('\\nTop 10 Bridge Patients (Betweenness Centrality):')\n"
         "for pid, bc in top_between:\n"
         "    grade = G.nodes[pid].get('dr_grade', '?')\n"
         "    print(f'  {pid}: betweenness={bc:.4f}  DR_grade={grade}')"),

    md("## 5. DR Grade Homophily"),

    code("analyzer = NetworkAnalyzer(\n"
         "    clinical_df=clinical_df,\n"
         "    feature_columns=feature_cols,\n"
         "    threshold=cfg['network']['edge_threshold'],\n"
         "    dr_grades=dr_grades,\n"
         ")\n"
         "homophily = analyzer.compute_dr_homophily()\n"
         "print(f'DR Grade Homophily: {homophily:.4f}')\n"
         "print('  1.0 = all edges connect same-grade patients')\n"
         "print('  0.2 = random (expected if no structure)')"),

    md("## 6. Network Visualizations"),

    code("Path('../outputs/network').mkdir(parents=True, exist_ok=True)\n\n"
         "# Community view\n"
         "analyzer.visualize_network(\n"
         "    save_path='../outputs/network/patient_network_community.png',\n"
         "    color_by='community', size_by='degree', layout='spring',\n"
         ")\n\n"
         "# DR grade view\n"
         "analyzer.visualize_network(\n"
         "    save_path='../outputs/network/patient_network_drgrade.png',\n"
         "    color_by='dr_grade', size_by='betweenness', layout='spring',\n"
         ")\n\n"
         "fig, axes = plt.subplots(1, 2, figsize=(18, 8))\n"
         "for ax, fname, title in zip(axes,\n"
         "    ['../outputs/network/patient_network_community.png',\n"
         "     '../outputs/network/patient_network_drgrade.png'],\n"
         "    ['Coloured by Community', 'Coloured by DR Grade']):\n"
         "    import cv2\n"
         "    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)\n"
         "    ax.imshow(img)\n"
         "    ax.set_title(title, fontsize=13, fontweight='bold')\n"
         "    ax.axis('off')\n"
         "plt.tight_layout()\nplt.show()"),

    md("## 7. High-Risk Cluster Identification"),

    code("high_risk = analyzer.get_high_risk_clusters()\n"
         "print('High-risk communities (sorted by mean DR grade):\\n')\n"
         "for comm_id, stats in list(high_risk.items())[:8]:\n"
         "    print(f'  Community {comm_id}:')\n"
         "    print(f'    Size: {stats[\"size\"]} patients')\n"
         "    print(f'    Mean DR grade: {stats[\"mean_dr_grade\"]:.2f}')\n"
         "    print(f'    Severe fraction: {stats[\"severe_fraction\"]:.1%}')"),

    code("# Gephi export\nanalyzer.export_gexf('../outputs/network/patient_graph.gexf')\n"
         "print('GEXF exported — open in Gephi for interactive exploration')\n"
         "print('Tip: In Gephi: Layout > ForceAtlas2, Appearance > Nodes > Colour by community')"),

    md("## Summary\n\n"
       "| Metric | Value | Interpretation |\n"
       "|---|---|---|\n"
       "| Modularity Q | *computed above* | > 0.3 = meaningful clusters |\n"
       "| DR Homophily | *computed above* | > 0.5 = grade-based clustering |\n"
       "| High-risk clusters | *computed above* | Communities with mean grade ≥ 3 |"),
])

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 06 — Integration & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
nb06 = nb([
    md("# 06 — Integration & Evaluation\n\n"
       "This notebook ties all three modules together and produces the final evaluation:\n\n"
       "1. Load trained ResNet-50 and run inference on test set\n"
       "2. Full evaluation: QWK, AUC-ROC, F1, confusion matrix\n"
       "3. Baseline comparison (ResNet-50 vs VGG-16 vs Scratch)\n"
       "4. Attach CNN predictions to patient network\n"
       "5. Generate final network visualisation with CNN-predicted DR grades\n"
       "6. Produce results summary table"),

    code("import sys\nfrom pathlib import Path\n"
         "sys.path.insert(0, str(Path('..').resolve()))\n\n"
         "import numpy as np\nimport pandas as pd\nimport torch\nimport matplotlib.pyplot as plt\n\n"
         "from src.utils.config import load_config, get_device\n"
         "from src.models import build_resnet50\n"
         "from src.data import get_val_transforms\n"
         "from src.data.dataset import create_data_loaders\n"
         "from src.evaluation import compute_all_metrics, plot_confusion_matrix, plot_roc_curves\n\n"
         "cfg    = load_config('../configs/default.yaml')\n"
         "device = get_device()\n"
         "print(f'Device: {device}')"),

    md("## 1. Load Best Model & Inference"),

    code("model = build_resnet50(num_classes=5, fc_hidden=512, dropout=0.3,\n"
         "                        pretrained=False).to(device)\n"
         "ckpt = torch.load('../outputs/models/best_model.pth', map_location=device)\n"
         "model.load_state_dict(ckpt['model_state_dict'])\n"
         "model.eval()\n"
         "print(f'Loaded epoch {ckpt[\"epoch\"]} | Val QWK: {ckpt[\"val_qwk\"]:.4f}')"),

    code("_, _, test_loader = create_data_loaders(\n"
         "    train_csv=cfg['paths']['train_csv'],\n"
         "    val_csv=cfg['paths']['val_csv'],\n"
         "    test_csv=cfg['paths']['test_csv'],\n"
         "    train_image_dir=cfg['paths']['train_images'],\n"
         "    val_image_dir=cfg['paths']['val_images'],\n"
         "    test_image_dir=cfg['paths']['test_images'],\n"
         "    train_transform=get_val_transforms(224),\n"
         "    val_transform=get_val_transforms(224),\n"
         "    image_size=224, batch_size=32,\n"
         ")\n\n"
         "all_preds, all_labels, all_probs = [], [], []\n"
         "with torch.no_grad():\n"
         "    for images, labels in test_loader:\n"
         "        images = images.to(device)\n"
         "        logits = model(images)\n"
         "        probs  = torch.softmax(logits, dim=1)\n"
         "        preds  = logits.argmax(dim=1)\n"
         "        all_preds.extend(preds.cpu().numpy())\n"
         "        all_labels.extend(labels.numpy())\n"
         "        all_probs.extend(probs.cpu().numpy())\n\n"
         "all_preds  = np.array(all_preds)\n"
         "all_labels = np.array(all_labels)\n"
         "all_probs  = np.array(all_probs)\n"
         "print(f'Inference complete on {len(all_labels)} test images')"),

    md("## 2. Full Evaluation"),

    code("metrics = compute_all_metrics(all_labels, all_preds, all_probs)\n\n"
         "print('=' * 50)\n"
         "print('  ResNet-50 Test Set Results')\n"
         "print('=' * 50)\n"
         "for k, v in metrics.items():\n"
         "    if v is not None:\n"
         "        print(f'  {k:30s}: {v:.4f}')"),

    code("plot_confusion_matrix(all_labels, all_preds,\n"
         "                       '../outputs/figures/confusion_matrix.png')\n"
         "plot_roc_curves(all_labels, all_probs,\n"
         "                 '../outputs/figures/roc_curves.png')\n\n"
         "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
         "import cv2\n"
         "for ax, fname, title in zip(axes,\n"
         "    ['../outputs/figures/confusion_matrix.png',\n"
         "     '../outputs/figures/roc_curves.png'],\n"
         "    ['Confusion Matrix', 'ROC Curves']):\n"
         "    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)\n"
         "    ax.imshow(img); ax.set_title(title, fontsize=13, fontweight='bold'); ax.axis('off')\n"
         "plt.tight_layout()\nplt.show()"),

    md("## 3. Baseline Comparison"),

    code("# Placeholder — fill in after training VGG-16 and scratch model in Notebook 03\n"
         "results = {\n"
         "    'ResNet-50 (ours)': metrics,\n"
         "    # 'VGG-16': metrics_vgg,\n"
         "    # 'Scratch':  metrics_scratch,\n"
         "}\n\n"
         "summary_rows = []\n"
         "for model_name, m in results.items():\n"
         "    summary_rows.append({\n"
         "        'Model': model_name,\n"
         "        'QWK ↑': round(m.get('quadratic_weighted_kappa', 0), 4),\n"
         "        'AUC-ROC ↑': round(m.get('auroc', 0), 4),\n"
         "        'F1 Weighted ↑': round(m.get('f1_weighted', 0), 4),\n"
         "        'Accuracy ↑': round(m.get('accuracy', 0), 4),\n"
         "    })\n\n"
         "summary_df = pd.DataFrame(summary_rows)\n"
         "summary_df.to_csv('../outputs/results/baseline_comparison.csv', index=False)\n"
         "print(summary_df.to_string(index=False))"),

    md("## 4. Integrate CNN Predictions into Patient Network"),

    code("from src.network.analysis import NetworkAnalyzer\n"
         "from src.utils.config import load_config\n\n"
         "clinical_df = pd.read_csv('../data/metadata/clinical_metadata.csv')\n"
         "feature_cols = [c for c in cfg['network']['clinical_features']\n"
         "                if c in clinical_df.columns]\n\n"
         "# For patients in metadata, use CNN-predicted grades\n"
         "# (here we use the stored dr_grade as a proxy; in production,\n"
         "#  map patient_id → CNN prediction)\n"
         "dr_grades = clinical_df['dr_grade'].values if 'dr_grade' in clinical_df.columns else None\n\n"
         "analyzer = NetworkAnalyzer(\n"
         "    clinical_df=clinical_df,\n"
         "    feature_columns=feature_cols,\n"
         "    threshold=cfg['network']['edge_threshold'],\n"
         "    dr_grades=dr_grades,\n"
         ")\n"
         "summary = analyzer.summary()\n"
         "print('Network Summary:')\n"
         "for k, v in summary.items():\n"
         "    print(f'  {k}: {v}')"),

    md("## 5. Final Results Summary"),

    code("print('\\n' + '=' * 60)\n"
         "print('  Praxis — Final Results Summary')\n"
         "print('=' * 60)\n\n"
         "print('\\n  CNN Performance (Test Set):')\n"
         "for k, v in metrics.items():\n"
         "    if v is not None:\n"
         "        print(f'    {k}: {v:.4f}')\n\n"
         "print('\\n  Patient Network:')\n"
         "for k, v in summary.items():\n"
         "    if v is not None:\n"
         "        val_str = f'{v:.4f}' if isinstance(v, float) else str(v)\n"
         "        print(f'    {k}: {val_str}')\n\n"
         "print('\\n  Outputs saved to outputs/ directory')\n"
         "print('  Run the Streamlit dashboard:  streamlit run app/streamlit_app.py')"),
])


# ═══════════════════════════════════════════════════════════════════════════════
# Write all notebooks
# ═══════════════════════════════════════════════════════════════════════════════
NOTEBOOK_FILES = {
    "01_EDA.ipynb": nb01,
    "02_Preprocessing.ipynb": nb02,
    "03_Model_Training.ipynb": nb03,
    "04_GradCAM.ipynb": nb04,
    "05_Network_Analysis.ipynb": nb05,
    "06_Integration_Evaluation.ipynb": nb06,
}

if __name__ == "__main__":
    for fname, notebook in NOTEBOOK_FILES.items():
        out_path = NOTEBOOKS_DIR / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"[OK] Created: {out_path}")

    print(f"\nAll 6 notebooks saved to: {NOTEBOOKS_DIR}")
