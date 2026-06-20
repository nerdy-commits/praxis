# Explainable Diabetic Retinopathy Grading with Patient Comorbidity Risk Networks
## Complete Technical Blueprint & Execution Roadmap
### IIIT Kottayam — Data Science Bootcamp 2026 | Praxis Module

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack Specification](#3-tech-stack-specification)
4. [Dataset Sources & Acquisition Guide](#4-dataset-sources--acquisition-guide)
5. [Directory Structure](#5-directory-structure)
6. [Day-by-Day Execution Roadmap](#6-day-by-day-execution-roadmap)
7. [Phase A: Data Pipeline](#7-phase-a-data-pipeline)
8. [Phase B: CNN Model](#8-phase-b-cnn-model)
9. [Phase C: Explainability (Grad-CAM)](#9-phase-c-explainability-grad-cam)
10. [Phase D: Network Science Module](#10-phase-d-network-science-module)
11. [Phase E: Evaluation Framework](#11-phase-e-evaluation-framework)
12. [Phase F: Integration & Presentation](#12-phase-f-integration--presentation)
13. [GitHub Repository Guide](#13-github-repository-guide)
14. [Risk Register & Contingencies](#14-risk-register--contingencies)
15. [Portfolio Maximization Checklist](#15-portfolio-maximization-checklist)

---

## 1. Project Overview

### Problem Statement
Diabetic Retinopathy (DR) is a microvascular complication of diabetes and the leading cause of preventable blindness worldwide. Manual grading by ophthalmologists is time-consuming and subject to inter-observer variability. This project builds:

1. **A fine-tuned CNN classifier** (ResNet-50) that grades DR severity (0–4) from fundus images
2. **A Grad-CAM explainability layer** that highlights lesion regions driving each prediction
3. **A patient comorbidity similarity network** that maps high-risk patient clusters using clinical metadata and CNN-predicted DR grades as node attributes

### Research Contribution Angle
> "Combining CNN-based DR grading with graph-based patient stratification to identify comorbidity-driven high-risk clusters" — this cross-modal integration is non-standard in deployed systems and publishable at ML-in-Healthcare workshop tracks (MICCAI, NeurIPS HealthAI, ICLR workshops).

### Programme Outcomes Addressed
| PO | How Addressed |
|---|---|
| PO1 | End-to-end pipeline from raw images to final dashboard |
| PO2 | ML applied to ophthalmology/healthcare with rigorous validation |
| PO3 | CNN architecture design + transfer learning pipeline |
| PO4 | Patient similarity graph + centrality analysis |
| PO5 | Collaborative final presentation with validated metrics |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                   │
│   Fundus Images (APTOS 2019)    Clinical Metadata (Pima/Synthetic)   │
└────────────────┬──────────────────────────┬─────────────────────────┘
                 │                          │
                 ▼                          ▼
┌───────────────────────┐      ┌────────────────────────────┐
│   PREPROCESSING       │      │   METADATA PROCESSING      │
│  Ben Graham Filter    │      │   Normalization + Encoding │
│  Resize 224×224       │      │   Feature Vector Assembly  │
│  Augmentation         │      └────────────┬───────────────┘
│  Train/Val/Test Split │                   │
└──────────┬────────────┘                   │
           │                                │
           ▼                                │
┌───────────────────────┐                   │
│   CNN MODULE          │                   │
│   ResNet-50 Backbone  │                   │
│   (ImageNet Pretrain) │                   │
│   Fine-tuned FC Head  │                   │
│   Output: DR Grade    │◄──────────────────┘
│   (Softmax, 5-class)  │         (merged at integration)
└──────────┬────────────┘
           │
           ├──────────────────────────────────────┐
           ▼                                      ▼
┌──────────────────────┐           ┌──────────────────────────────┐
│   EXPLAINABILITY     │           │   NETWORK SCIENCE MODULE     │
│   Grad-CAM           │           │   Patient Similarity Graph   │
│   Guided Backprop    │           │   Cosine Similarity Edges    │
│   Heatmap Overlay    │           │   Louvain Clustering         │
└──────────────────────┘           │   Centrality Analysis        │
                                   │   DR Grade Node Attributes   │
                                   └──────────────┬───────────────┘
                                                  │
                                                  ▼
                              ┌──────────────────────────────────┐
                              │   EVALUATION & VISUALIZATION     │
                              │   QWK, AUC-ROC, F1, CM          │
                              │   Network Stats + Gephi Export   │
                              │   Streamlit Dashboard            │
                              └──────────────────────────────────┘
```

---

## 3. Tech Stack Specification

### 3.1 Core Environment

| Component | Tool | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | All modules |
| Environment Manager | conda / venv | latest | Dependency isolation |
| Notebook | Jupyter Lab | 4.x | Exploration & EDA |
| Version Control | Git + GitHub | latest | Portfolio + reproducibility |
| Experiment Tracking | MLflow (optional) | 2.x | Run tracking |

### 3.2 Deep Learning Stack

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.2.x | Model training, backprop |
| torchvision | 0.17.x | ResNet-50 pretrained weights, transforms |
| timm | 0.9.x | Alternative pretrained model zoo |
| albumentations | 1.3.x | Medical image augmentation |
| Pillow (PIL) | 10.x | Image I/O |
| OpenCV (cv2) | 4.9.x | Ben Graham preprocessing |
| pytorch-grad-cam | 1.5.x | Grad-CAM, EigenCAM, ScoreCAM |

### 3.3 Data Science & ML Stack

| Library | Version | Purpose |
|---|---|---|
| numpy | 1.26.x | Numerical computation |
| pandas | 2.2.x | Metadata handling |
| scikit-learn | 1.4.x | Metrics, preprocessing, baselines |
| scipy | 1.12.x | Statistical functions |
| imbalanced-learn | 0.12.x | Class imbalance handling |

### 3.4 Network Science Stack

| Library | Version | Purpose |
|---|---|---|
| networkx | 3.2.x | Graph construction & analysis |
| python-louvain | 0.16 | Community detection (Louvain) |
| pyvis | 0.3.x | Interactive network visualization |
| gephi | 0.10.x | Desktop: advanced network visualization |

### 3.5 Visualization Stack

| Library | Version | Purpose |
|---|---|---|
| matplotlib | 3.8.x | Base plotting |
| seaborn | 0.13.x | Statistical plots |
| plotly | 5.18.x | Interactive charts |
| streamlit | 1.32.x | Final demo dashboard |

### 3.6 Evaluation Stack

| Library | Version | Purpose |
|---|---|---|
| scikit-learn | 1.4.x | F1, AUC-ROC, confusion matrix |
| lifelines | 0.28.x | (Optional) Survival analysis for Module 02 crossover |
| scipy.stats | 1.12.x | Cohen's Kappa → QWK computation |

### 3.7 Environment Setup Script

```bash
# Create conda environment
conda create -n dr_project python=3.10 -y
conda activate dr_project

# Core ML
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm albumentations opencv-python-headless pillow

# Grad-CAM
pip install grad-cam

# Data Science
pip install numpy pandas scikit-learn scipy imbalanced-learn

# Network Science
pip install networkx python-louvain pyvis

# Visualization
pip install matplotlib seaborn plotly streamlit

# Utilities
pip install tqdm mlflow jupyter jupyterlab

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. Dataset Sources & Acquisition Guide

### 4.1 Primary Dataset: APTOS 2019 Blindness Detection

| Attribute | Detail |
|---|---|
| Source | Kaggle Competition |
| URL | https://www.kaggle.com/c/aptos2019-blindness-detection |
| Images | 3,662 fundus photographs |
| Labels | 5 classes: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR |
| Format | PNG, variable resolution (~2000×2000 px) |
| License | Competition use permitted |

**Class Distribution (approximate):**
```
Class 0 (No DR)         : ~1,805  (49.3%)  ← MAJORITY
Class 1 (Mild)          :   370  (10.1%)
Class 2 (Moderate)      :   999  (27.3%)
Class 3 (Severe)        :   193   (5.3%)
Class 4 (Proliferative) :   295   (8.1%)  ← Severe imbalance
```

**Download via Kaggle API:**
```bash
# Install Kaggle CLI
pip install kaggle

# Place kaggle.json at ~/.kaggle/kaggle.json
# Get from: Kaggle → Account → Create API Token

kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d data/raw/
```

### 4.2 Supplementary Dataset: Clinical Metadata

**Option A — Pima Indians Diabetes Dataset (Primary Recommendation)**
```
Source  : UCI ML Repository / Kaggle
URL     : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
Records : 768 patients
Features: Pregnancies, Glucose, BloodPressure, SkinThickness, 
          Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
```

**Option B — Synthetic Metadata Generation** (if dataset mapping is complex)

Generate realistic clinical metadata mapped to APTOS patient IDs:
```python
import numpy as np
import pandas as pd

def generate_clinical_metadata(n_patients, dr_grades, seed=42):
    """
    Generate synthetic but clinically plausible patient metadata.
    Clinical correlations:
    - Higher DR grade → higher HbA1c, longer diabetes duration
    - DR grade 3-4 → higher probability of hypertension, nephropathy
    """
    np.random.seed(seed)
    
    metadata = []
    for i, grade in enumerate(dr_grades):
        # HbA1c increases with DR severity (normal < 5.7, diabetic > 6.5)
        hba1c_base = [7.2, 7.8, 8.5, 9.2, 10.1][grade]
        hba1c = np.random.normal(hba1c_base, 0.8)
        
        # Diabetes duration (years)
        duration_base = [4, 6, 9, 13, 16][grade]
        duration = max(1, np.random.normal(duration_base, 2))
        
        # Comorbidities (binary flags)
        hypertension_prob = [0.25, 0.35, 0.50, 0.65, 0.75][grade]
        nephropathy_prob  = [0.05, 0.10, 0.20, 0.40, 0.55][grade]
        neuropathy_prob   = [0.10, 0.15, 0.28, 0.45, 0.60][grade]
        
        metadata.append({
            'patient_id'   : f'PT_{i:04d}',
            'dr_grade'     : grade,
            'hba1c'        : round(np.clip(hba1c, 5.5, 14.0), 2),
            'diabetes_dur' : round(duration, 1),
            'bmi'          : round(np.random.normal(27.5, 4.5), 1),
            'systolic_bp'  : int(np.random.normal(130 + 10*grade, 15)),
            'age'          : int(np.random.normal(52 + 2*grade, 10)),
            'hypertension' : int(np.random.binomial(1, hypertension_prob)),
            'nephropathy'  : int(np.random.binomial(1, nephropathy_prob)),
            'neuropathy'   : int(np.random.binomial(1, neuropathy_prob)),
        })
    
    return pd.DataFrame(metadata)
```

### 4.3 Supplementary Reference Datasets

| Dataset | Use Case | Source |
|---|---|---|
| Messidor-2 | External validation of CNN | https://www.adcis.net/en/third-party/messidor2/ |
| IDRiD | Lesion-level segmentation reference | https://idrid.grand-challenge.org/ |
| EyePACS (subset) | Scale testing | Kaggle DR 2015 competition |

---

## 5. Directory Structure

```
dr_retinopathy_project/
│
├── data/
│   ├── raw/
│   │   ├── train_images/           # APTOS raw PNG files
│   │   ├── train.csv               # patient_id + diagnosis label
│   │   └── test_images/
│   ├── processed/
│   │   ├── train/                  # Ben Graham preprocessed + resized
│   │   │   ├── 0/                  # Organized by class for ImageFolder
│   │   │   ├── 1/
│   │   │   ├── 2/
│   │   │   ├── 3/
│   │   │   └── 4/
│   │   ├── val/
│   │   └── test/
│   └── metadata/
│       ├── clinical_metadata.csv
│       └── patient_feature_vectors.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_GradCAM.ipynb
│   ├── 05_Network_Analysis.ipynb
│   └── 06_Integration_Evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py        # Ben Graham + augmentation
│   │   ├── dataset.py              # PyTorch Dataset class
│   │   └── metadata.py             # Clinical data processing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet_model.py         # ResNet-50 fine-tuning
│   │   └── baseline_models.py      # VGG-16, from-scratch baselines
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── gradcam.py              # Grad-CAM wrapper
│   ├── network/
│   │   ├── __init__.py
│   │   ├── graph_builder.py        # Patient similarity graph
│   │   └── graph_analysis.py       # Centrality, clustering, homophily
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py              # QWK, AUC-ROC, confusion matrix
│
├── outputs/
│   ├── models/                     # Saved .pth checkpoints
│   ├── gradcam/                    # Heatmap images
│   ├── network/                    # Graph files (.graphml, .gexf for Gephi)
│   ├── figures/                    # All publication-ready plots
│   └── results/                    # Metric tables (.csv)
│
├── app/
│   └── streamlit_app.py            # Final demo dashboard
│
├── requirements.txt
├── README.md                       # Comprehensive project README
└── .gitignore
```

---

## 6. Day-by-Day Execution Roadmap

```
WEEK 1 (Online): Foundation + Core Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 1  │ Setup + Data Acquisition + EDA
Day 2  │ Preprocessing Pipeline Implementation
Day 3  │ Dataset Class + DataLoaders + Augmentation Verification
Day 4  │ ResNet-50 Baseline Training (frozen backbone)
Day 5  │ Full Fine-tuning + Hyperparameter Tuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WEEK 2 (Online → Offline transition): XAI + Network Science
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 6  │ Grad-CAM Implementation + Visual Validation
Day 7  │ Clinical Metadata Preparation + Feature Engineering
Day 8  │ Patient Similarity Network Construction
Day 9  │ Graph Analysis (Centrality + Clustering)
Day 10 │ Integration: CNN Outputs → Network Node Attributes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WEEK 3 (Offline): Capstone + Evaluation + Presentation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 11 │ Comprehensive Evaluation + Baseline Comparisons
Day 12 │ Streamlit Dashboard Development
Day 13 │ Results Synthesis + README + Report
Day 14 │ Capstone Presentation (Dry Run)
Day 15 │ Final Presentation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 7. Phase A: Data Pipeline

### 7.1 Ben Graham Preprocessing

Ben Graham's method (winner of Kaggle DR 2015) enhances retinal vessel contrast by subtracting a blurred version of the image:

```python
# src/data/preprocessing.py
import cv2
import numpy as np
from pathlib import Path

def ben_graham_preprocess(img: np.ndarray, 
                           sigmaX: int = 10,
                           target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Ben Graham retinal image preprocessing.
    Steps:
      1. Resize to target
      2. Gaussian blur with large sigma
      3. Weighted addition: original * 4 - blurred * 4 + 128
      4. Circular crop to remove black border
    
    Args:
        img: Input BGR image (OpenCV format)
        sigmaX: Gaussian blur sigma (default 10)
        target_size: Output spatial dimensions
    Returns:
        Preprocessed image as uint8 numpy array
    """
    img = cv2.resize(img, target_size)
    img = cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), sigmaX), -4,
        128
    )
    # Circular mask to remove scan border artifacts
    mask = np.zeros(img.shape)
    cv2.circle(
        mask,
        center=(target_size[0]//2, target_size[1]//2),
        radius=int(target_size[0] * 0.45),
        color=(1, 1, 1),
        thickness=-1
    )
    img = (img * mask).astype(np.uint8)
    return img


def preprocess_dataset(raw_dir: str, 
                        output_dir: str,
                        labels_csv: str):
    """
    Batch preprocess all APTOS images and organize by class.
    """
    import pandas as pd
    from tqdm import tqdm
    
    df = pd.read_csv(labels_csv)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(raw_dir) / f"{row['id_code']}.png"
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        processed = ben_graham_preprocess(img)
        
        # Save organized by class (for ImageFolder compatibility)
        out_dir = Path(output_dir) / str(row['diagnosis'])
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{row['id_code']}.png"), processed)
```

### 7.2 PyTorch Dataset & DataLoaders

```python
# src/data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

# ImageNet normalization stats (required for ResNet-50 pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(mode: str = 'train'):
    """
    Albumentations-based transforms.
    Training: aggressive augmentation for medical images.
    Val/Test: only normalization.
    """
    if mode == 'train':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, 
                rotate_limit=30, p=0.7
            ),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, 
                    saturation=0.2, hue=0.1, p=1.0
                ),
            ], p=0.5),
            A.CoarseDropout(
                max_holes=8, max_height=16, max_width=16, p=0.3
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class APTOSDataset(Dataset):
    """
    Custom Dataset for APTOS 2019 with albumentations transforms.
    """
    def __init__(self, image_dir: str, labels_csv: str, 
                 transform=None, mode: str = 'train'):
        import pandas as pd
        from pathlib import Path
        
        self.df = pd.read_csv(labels_csv)
        self.image_dir = Path(image_dir)
        self.transform = transform or get_transforms(mode)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / f"{row['id_code']}.png"
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=img)
        image = augmented['image']
        label = torch.tensor(row['diagnosis'], dtype=torch.long)
        
        return image, label, row['id_code']


def get_dataloaders(data_dir: str, labels_csv: str, 
                    batch_size: int = 32,
                    num_workers: int = 4,
                    val_split: float = 0.15,
                    test_split: float = 0.15,
                    seed: int = 42):
    """
    Returns train, val, test DataLoaders with stratified splits.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(labels_csv)
    
    # Stratified split to preserve class distribution
    train_val, test = train_test_split(
        df, test_size=test_split, stratify=df['diagnosis'], 
        random_state=seed
    )
    train, val = train_test_split(
        train_val, 
        test_size=val_split / (1 - test_split),
        stratify=train_val['diagnosis'], 
        random_state=seed
    )
    
    # Save splits
    train.to_csv('data/splits/train.csv', index=False)
    val.to_csv('data/splits/val.csv', index=False)
    test.to_csv('data/splits/test.csv', index=False)
    
    datasets = {
        'train': APTOSDataset(data_dir, 'data/splits/train.csv', mode='train'),
        'val'  : APTOSDataset(data_dir, 'data/splits/val.csv',   mode='val'),
        'test' : APTOSDataset(data_dir, 'data/splits/test.csv',  mode='test'),
    }
    
    loaders = {
        split: DataLoader(
            ds, 
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        for split, ds in datasets.items()
    }
    
    return loaders
```

---

## 8. Phase B: CNN Model

### 8.1 ResNet-50 Fine-Tuning Architecture

```python
# src/models/resnet_model.py
import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple

class DRResNet50(nn.Module):
    """
    ResNet-50 fine-tuned for 5-class Diabetic Retinopathy grading.
    
    Strategy:
      Phase 1 (Days 3-4): Freeze backbone, train FC head only
      Phase 2 (Day 5):    Unfreeze last 2 ResNet blocks + FC head
    
    Architecture:
      ResNet-50 conv layers → AdaptiveAvgPool2d → 
      Dropout(0.5) → Linear(2048→512) → 
      ReLU → Dropout(0.3) → Linear(512→5)
    """
    
    def __init__(self, num_classes: int = 5, 
                 dropout1: float = 0.5,
                 dropout2: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        
        # Load pretrained backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        
        # Remove original FC layer
        # Output of avgpool: [batch, 2048, 1, 1]
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout1),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout2),
            nn.Linear(512, num_classes),
        )
        
        # Initialize custom FC layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)    # [B, 2048, 1, 1]
        logits   = self.classifier(features)  # [B, 5]
        return logits
    
    def freeze_backbone(self):
        """Phase 1: Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_blocks(self, n: int = 2):
        """Phase 2: Unfreeze last n ResNet blocks (layer3, layer4)."""
        # ResNet-50 blocks: layer1, layer2, layer3, layer4
        block_names = [f'layer{i}' for i in range(5-n, 5)]
        
        for name, module in self.backbone.named_modules():
            for block in block_names:
                if block in name:
                    for param in module.parameters():
                        param.requires_grad = True
```

### 8.2 Training Loop

```python
# src/models/trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score

def compute_qwk(y_true, y_pred):
    """Quadratic Weighted Kappa — official APTOS metric."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def get_class_weights(labels_csv: str, device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to handle imbalance.
    DR has heavy class 0 overrepresentation.
    """
    import pandas as pd
    df = pd.read_csv(labels_csv)
    counts = df['diagnosis'].value_counts().sort_index().values
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.FloatTensor(weights).to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, labels, _ in tqdm(loader, desc='Training'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * images.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    qwk = compute_qwk(all_labels, all_preds)
    return avg_loss, qwk


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, labels, _ in tqdm(loader, desc='Validation'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(images)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * images.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    qwk = compute_qwk(all_labels, all_preds)
    return avg_loss, qwk, np.array(all_labels), np.array(all_preds)


def train_pipeline(model, loaders, config: dict, device):
    """
    Complete two-phase training pipeline.
    
    Phase 1 (epochs 1–10):  Frozen backbone, high LR for FC
    Phase 2 (epochs 11–30): Unfrozen last 2 blocks, lower LR
    """
    criterion = nn.CrossEntropyLoss(
        weight=get_class_weights('data/splits/train.csv', device),
        label_smoothing=0.1  # Prevent overconfidence
    )
    
    best_qwk   = -1.0
    history    = {'train_loss': [], 'val_loss': [], 
                  'train_qwk': [],  'val_qwk': []}
    
    # ── PHASE 1: Train FC head only ─────────────────────────────────
    print("\n=== PHASE 1: Training FC Head ===")
    model.freeze_backbone()
    
    optimizer_p1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr_phase1'], weight_decay=1e-4
    )
    scheduler_p1 = CosineAnnealingLR(
        optimizer_p1, T_max=config['epochs_phase1']
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for epoch in range(config['epochs_phase1']):
        tr_loss, tr_qwk = train_one_epoch(
            model, loaders['train'], optimizer_p1, criterion, device, scaler
        )
        vl_loss, vl_qwk, _, _ = validate(
            model, loaders['val'], criterion, device
        )
        scheduler_p1.step()
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_qwk'].append(tr_qwk)
        history['val_qwk'].append(vl_qwk)
        
        print(f"Ep {epoch+1:02d} | "
              f"TL:{tr_loss:.4f} VL:{vl_loss:.4f} | "
              f"TQ:{tr_qwk:.4f} VQ:{vl_qwk:.4f}")
        
        if vl_qwk > best_qwk:
            best_qwk = vl_qwk
            torch.save(model.state_dict(), 'outputs/models/best_model.pth')
            print(f"  ✓ New best QWK: {best_qwk:.4f} — Model saved")
    
    # ── PHASE 2: Fine-tune last 2 blocks + FC ───────────────────────
    print("\n=== PHASE 2: Full Fine-tuning ===")
    model.unfreeze_last_n_blocks(n=2)
    
    optimizer_p2 = AdamW([
        {'params': model.backbone.parameters(), 'lr': config['lr_backbone']},
        {'params': model.classifier.parameters(), 'lr': config['lr_phase1']},
    ], weight_decay=1e-4)
    scheduler_p2 = OneCycleLR(
        optimizer_p2, 
        max_lr=[config['lr_backbone'], config['lr_phase1']],
        epochs=config['epochs_phase2'],
        steps_per_epoch=len(loaders['train'])
    )
    
    for epoch in range(config['epochs_phase2']):
        tr_loss, tr_qwk = train_one_epoch(
            model, loaders['train'], optimizer_p2, criterion, device, scaler
        )
        vl_loss, vl_qwk, _, _ = validate(
            model, loaders['val'], criterion, device
        )
        scheduler_p2.step()
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_qwk'].append(tr_qwk)
        history['val_qwk'].append(vl_qwk)
        
        print(f"Ep {config['epochs_phase1']+epoch+1:02d} | "
              f"TL:{tr_loss:.4f} VL:{vl_loss:.4f} | "
              f"TQ:{tr_qwk:.4f} VQ:{vl_qwk:.4f}")
        
        if vl_qwk > best_qwk:
            best_qwk = vl_qwk
            torch.save(model.state_dict(), 'outputs/models/best_model.pth')
            print(f"  ✓ New best QWK: {best_qwk:.4f} — Model saved")
    
    return model, history


# Training configuration
TRAIN_CONFIG = {
    'epochs_phase1' : 10,
    'epochs_phase2' : 20,
    'lr_phase1'     : 3e-4,   # FC head learning rate
    'lr_backbone'   : 5e-5,   # Fine-tuning backbone LR (10x smaller)
    'batch_size'    : 32,
}
```

---

## 9. Phase C: Explainability (Grad-CAM)

### 9.1 Grad-CAM Implementation

```python
# src/explainability/gradcam.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_gradcam_visualizer(model, use_cuda: bool = True):
    """
    Set up Grad-CAM on the last conv layer of ResNet-50.
    Target layer: layer4[-1] → the final residual block.
    """
    target_layers = [model.backbone[-2][-1]]  # layer4, last bottleneck
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    return cam


def generate_gradcam(model, image_tensor: torch.Tensor, 
                     original_image: np.ndarray,
                     predicted_class: int,
                     save_path: str = None) -> np.ndarray:
    """
    Generate and optionally save Grad-CAM heatmap overlay.
    
    Args:
        model: Trained DRResNet50
        image_tensor: Preprocessed tensor [1, 3, 224, 224]
        original_image: Normalized float32 image [224, 224, 3] in [0,1]
        predicted_class: Model's predicted DR grade
        save_path: If provided, saves the visualization
    
    Returns:
        cam_image: Heatmap overlaid on original image
    """
    cam = get_gradcam_visualizer(model)
    targets = [ClassifierOutputTarget(predicted_class)]
    
    grayscale_cam = cam(
        input_tensor=image_tensor.unsqueeze(0),
        targets=targets
    )
    grayscale_cam = grayscale_cam[0]  # Remove batch dim
    
    # Overlay heatmap on original image
    cam_image = show_cam_on_image(
        original_image.astype(np.float32) / 255.0,
        grayscale_cam,
        use_rgb=True,
        colormap=cv2.COLORMAP_JET
    )
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    
    return cam_image


def batch_gradcam_analysis(model, test_loader, device, 
                            n_samples: int = 20,
                            output_dir: str = 'outputs/gradcam/'):
    """
    Generate Grad-CAM for representative samples across all DR grades.
    Selects n_samples/5 images per class for balanced visualization.
    """
    import os
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    
    grade_labels = {
        0: 'No_DR', 1: 'Mild', 2: 'Moderate', 
        3: 'Severe', 4: 'Proliferative_DR'
    }
    
    collected = {i: [] for i in range(5)}
    per_class = n_samples // 5
    
    with torch.no_grad():
        for images, labels, ids in test_loader:
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            
            for i, (img, label, pred, img_id) in enumerate(
                zip(images, labels, preds, ids)
            ):
                cls = label.item()
                if len(collected[cls]) < per_class:
                    collected[cls].append({
                        'tensor': img.cpu(),
                        'label' : cls,
                        'pred'  : pred.item(),
                        'id'    : img_id,
                        'correct': cls == pred.item()
                    })
            
            if all(len(v) >= per_class for v in collected.values()):
                break
    
    # Generate and save Grad-CAMs
    for cls, samples in collected.items():
        for s in samples:
            # Convert tensor back to displayable image
            img_np = s['tensor'].permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_np = (img_np * 255).astype(np.uint8)
            
            cam_img = generate_gradcam(
                model=model,
                image_tensor=s['tensor'].to(device),
                original_image=img_np,
                predicted_class=s['pred'],
                save_path=os.path.join(
                    output_dir,
                    f"{grade_labels[cls]}_pred{s['pred']}"
                    f"_{'correct' if s['correct'] else 'wrong'}"
                    f"_{s['id']}.png"
                )
            )
    
    print(f"Grad-CAM images saved to {output_dir}")
```

---

## 10. Phase D: Network Science Module

### 10.1 Patient Similarity Graph Construction

```python
# src/network/graph_builder.py
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def build_feature_vectors(metadata_df: pd.DataFrame) -> np.ndarray:
    """
    Construct normalized feature vectors for each patient.
    Features used for similarity computation:
      - HbA1c (continuous)
      - Diabetes duration (continuous)
      - BMI (continuous)
      - Systolic BP (continuous)
      - Age (continuous)
      - Hypertension (binary)
      - Nephropathy (binary)
      - Neuropathy (binary)
    """
    continuous_features = ['hba1c', 'diabetes_dur', 'bmi', 'systolic_bp', 'age']
    binary_features     = ['hypertension', 'nephropathy', 'neuropathy']
    
    # Normalize continuous features to [0, 1]
    scaler = MinMaxScaler()
    cont_scaled = scaler.fit_transform(metadata_df[continuous_features])
    
    # Binary features already in {0, 1}
    binary_vals = metadata_df[binary_features].values
    
    # Concatenate: [continuous(5) | binary(3)] → 8-dim feature vector
    feature_vectors = np.hstack([cont_scaled, binary_vals])
    
    return feature_vectors, scaler


def build_patient_similarity_graph(metadata_df: pd.DataFrame,
                                    cnn_predictions: dict,
                                    similarity_threshold: float = 0.85,
                                    max_edges_per_node: int = 10) -> nx.Graph:
    """
    Construct patient similarity network.
    
    Nodes  : Patients (identified by patient_id)
    Edges  : Cosine similarity between clinical feature vectors > threshold
    Attrs  : DR grade (ground truth + CNN prediction), community, centrality
    
    Args:
        metadata_df: DataFrame with clinical features
        cnn_predictions: dict {patient_id: {'true': int, 'pred': int, 'conf': float}}
        similarity_threshold: Minimum cosine similarity for edge creation
        max_edges_per_node: Limit edges to top-k most similar neighbors
    
    Returns:
        G: NetworkX Graph with node/edge attributes
    """
    feature_vectors, _ = build_feature_vectors(metadata_df)
    
    # Compute pairwise cosine similarity matrix
    sim_matrix = cosine_similarity(feature_vectors)  # [N, N]
    
    # Build graph
    G = nx.Graph()
    patient_ids = metadata_df['patient_id'].values
    
    # Add nodes with attributes
    for i, pid in enumerate(patient_ids):
        pred_info = cnn_predictions.get(pid, {})
        G.add_node(pid, 
            # Clinical attributes
            hba1c        = float(metadata_df.iloc[i]['hba1c']),
            diabetes_dur = float(metadata_df.iloc[i]['diabetes_dur']),
            hypertension = int(metadata_df.iloc[i]['hypertension']),
            # CNN attributes
            dr_grade_true = pred_info.get('true', -1),
            dr_grade_pred = pred_info.get('pred', -1),
            confidence    = pred_info.get('conf', 0.0),
        )
    
    # Add edges (with threshold + degree limit)
    for i in range(len(patient_ids)):
        # Get sorted neighbors by similarity (descending)
        sims = sim_matrix[i].copy()
        sims[i] = 0  # Exclude self-similarity
        
        top_k_idx = np.argsort(sims)[::-1][:max_edges_per_node]
        
        for j in top_k_idx:
            if sims[j] >= similarity_threshold:
                if not G.has_edge(patient_ids[i], patient_ids[j]):
                    G.add_edge(patient_ids[i], patient_ids[j],
                               weight=float(sims[j]))
    
    print(f"Graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, "
          f"Density: {nx.density(G):.4f}")
    
    return G
```

### 10.2 Graph Analysis

```python
# src/network/graph_analysis.py
import networkx as nx
import numpy as np
import pandas as pd
import community as community_louvain  # python-louvain
from collections import Counter

def analyze_graph(G: nx.Graph) -> dict:
    """
    Comprehensive network analysis:
      - Degree centrality
      - Betweenness centrality  
      - PageRank
      - Louvain community detection
      - DR grade homophily
      - High-risk cluster identification
    """
    results = {}
    
    # ── Centrality Measures ──────────────────────────────────────────
    results['degree_centrality']     = nx.degree_centrality(G)
    results['betweenness_centrality'] = nx.betweenness_centrality(G, normalized=True)
    results['pagerank']              = nx.pagerank(G, weight='weight', alpha=0.85)
    results['closeness_centrality']  = nx.closeness_centrality(G)
    
    # ── Community Detection (Louvain) ────────────────────────────────
    partition = community_louvain.best_partition(G, weight='weight')
    results['community'] = partition
    results['modularity'] = community_louvain.modularity(partition, G)
    print(f"Communities detected: {len(set(partition.values()))}")
    print(f"Modularity score: {results['modularity']:.4f}")
    
    # Add community as node attribute
    nx.set_node_attributes(G, partition, 'community')
    
    # Add centrality as node attributes
    nx.set_node_attributes(G, results['degree_centrality'],      'degree_centrality')
    nx.set_node_attributes(G, results['betweenness_centrality'],  'betweenness_centrality')
    nx.set_node_attributes(G, results['pagerank'],               'pagerank')
    
    # ── DR Grade Homophily ───────────────────────────────────────────
    # Measure: fraction of edges connecting same DR grade nodes
    same_grade_edges = 0
    total_edges = G.number_of_edges()
    
    for u, v in G.edges():
        if G.nodes[u]['dr_grade_pred'] == G.nodes[v]['dr_grade_pred']:
            same_grade_edges += 1
    
    homophily = same_grade_edges / total_edges if total_edges > 0 else 0
    results['dr_homophily'] = homophily
    print(f"DR Grade Homophily: {homophily:.4f} "
          f"(1.0=perfect clustering, 0.0=random)")
    
    # ── High-Risk Patient Identification ────────────────────────────
    # High-risk = high betweenness centrality + DR grade ≥ 3
    high_risk = [
        node for node, bc in results['betweenness_centrality'].items()
        if bc > np.percentile(list(results['betweenness_centrality'].values()), 90)
        and G.nodes[node]['dr_grade_pred'] >= 3
    ]
    results['high_risk_patients'] = high_risk
    print(f"High-risk bridge patients identified: {len(high_risk)}")
    
    # ── Community DR Grade Distribution ─────────────────────────────
    community_stats = {}
    for community_id in set(partition.values()):
        members = [n for n, c in partition.items() if c == community_id]
        grades  = [G.nodes[m]['dr_grade_pred'] for m in members 
                   if G.nodes[m]['dr_grade_pred'] >= 0]
        
        community_stats[community_id] = {
            'size': len(members),
            'avg_dr_grade': np.mean(grades) if grades else 0,
            'severe_count': sum(1 for g in grades if g >= 3),
            'grade_dist'  : Counter(grades)
        }
    
    results['community_stats'] = community_stats
    
    return results, G


def export_for_gephi(G: nx.Graph, output_path: str = 'outputs/network/patient_graph.gexf'):
    """Export graph in GEXF format for Gephi visualization."""
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, output_path)
    print(f"Graph exported for Gephi: {output_path}")
    

def export_for_pyvis(G: nx.Graph, output_path: str = 'outputs/network/patient_network.html'):
    """Interactive browser visualization using PyVis."""
    from pyvis.network import Network
    from pathlib import Path
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Color map for DR grades
    grade_colors = {
        0: '#2ecc71',  # Green: No DR
        1: '#f1c40f',  # Yellow: Mild
        2: '#e67e22',  # Orange: Moderate
        3: '#e74c3c',  # Red: Severe
        4: '#8e44ad',  # Purple: Proliferative
       -1: '#95a5a6',  # Grey: Unknown
    }
    
    net = Network(height='750px', width='100%', 
                  bgcolor='#1a1a2e', font_color='white')
    net.toggle_physics(True)
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    for node, data in G.nodes(data=True):
        grade = data.get('dr_grade_pred', -1)
        bc    = data.get('betweenness_centrality', 0.0)
        size  = 10 + bc * 100  # Scale node size by centrality
        
        net.add_node(
            node,
            label=f"P:{grade}",
            color=grade_colors.get(grade, '#95a5a6'),
            size=size,
            title=(f"ID: {node}\n"
                   f"DR Grade: {grade}\n"
                   f"HbA1c: {data.get('hba1c', 'N/A')}\n"
                   f"Betweenness: {bc:.4f}\n"
                   f"Community: {data.get('community', 'N/A')}")
        )
    
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, value=data.get('weight', 0.5))
    
    net.save_graph(output_path)
    print(f"Interactive network saved: {output_path}")
```

---

## 11. Phase E: Evaluation Framework

```python
# src/evaluation/metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score, roc_auc_score,
    f1_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

DR_GRADE_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

def evaluate_model(y_true: np.ndarray, 
                   y_pred: np.ndarray,
                   y_prob: np.ndarray,
                   model_name: str = 'ResNet-50',
                   save_dir: str = 'outputs/figures/') -> dict:
    """
    Complete evaluation suite for DR grading model.
    
    Returns:
        metrics: dict with QWK, macro AUC-ROC, per-class F1, accuracy
    """
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    metrics = {}
    
    # ── Primary Metric: Quadratic Weighted Kappa ─────────────────────
    metrics['qwk'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # ── AUC-ROC (One-vs-Rest, macro average) ─────────────────────────
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    metrics['auc_roc_macro'] = roc_auc_score(y_bin, y_prob, 
                                               multi_class='ovr', 
                                               average='macro')
    
    # ── Per-class AUC-ROC ────────────────────────────────────────────
    for i, grade in enumerate(DR_GRADE_NAMES):
        metrics[f'auc_{grade.replace(" ", "_")}'] = roc_auc_score(
            y_bin[:, i], y_prob[:, i]
        )
    
    # ── F1 Score ─────────────────────────────────────────────────────
    metrics['f1_macro']    = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    # ── Accuracy ─────────────────────────────────────────────────────
    metrics['accuracy'] = (y_true == y_pred).mean()
    
    # ── Classification Report ────────────────────────────────────────
    report = classification_report(y_true, y_pred, 
                                    target_names=DR_GRADE_NAMES)
    
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  QWK (primary):    {metrics['qwk']:.4f}")
    print(f"  Macro AUC-ROC:    {metrics['auc_roc_macro']:.4f}")
    print(f"  Macro F1:         {metrics['f1_macro']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"\n{report}")
    
    # ── Confusion Matrix Plot ────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DR_GRADE_NAMES,
                yticklabels=DR_GRADE_NAMES)
    plt.title(f'{model_name} — Confusion Matrix\nQWK = {metrics["qwk"]:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_confusion_matrix.png', dpi=150)
    plt.close()
    
    return metrics


def compare_baselines(results_dict: dict, save_dir: str = 'outputs/figures/'):
    """
    Compare multiple models side-by-side.
    results_dict = {'ResNet-50': metrics, 'VGG-16': metrics, 'Scratch': metrics}
    """
    comparison_df = pd.DataFrame(results_dict).T
    comparison_df = comparison_df[['qwk', 'auc_roc_macro', 'f1_macro', 'accuracy']]
    comparison_df.columns = ['QWK ↑', 'AUC-ROC ↑', 'F1 Macro ↑', 'Accuracy ↑']
    
    print("\n=== Baseline Comparison Table ===")
    print(comparison_df.round(4).to_string())
    comparison_df.to_csv(f'{save_dir}/baseline_comparison.csv')
    
    # Bar chart comparison
    ax = comparison_df.plot(kind='bar', figsize=(10, 5), 
                             colormap='viridis', edgecolor='white')
    ax.set_title('Model Comparison — Evaluation Metrics')
    ax.set_ylabel('Score')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/baseline_comparison.png', dpi=150)
    plt.close()
```

---

## 12. Phase F: Integration & Presentation

### 12.1 Streamlit Dashboard

```python
# app/streamlit_app.py
import streamlit as st
import torch
import numpy as np
import cv2
import networkx as nx
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title="DR Risk Stratification System",
    page_icon="🔬",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────
st.title("🔬 Diabetic Retinopathy Grading + Patient Risk Network")
st.markdown("**IIIT Kottayam | Data Science Bootcamp 2026 | Praxis Project**")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Module", [
    "📊 Project Overview",
    "🖼️ DR Grading (CNN)",
    "🔍 Explainability (Grad-CAM)",
    "🌐 Patient Risk Network",
    "📈 Evaluation Metrics"
])

# ── Page 2: DR Grading ────────────────────────────────────────────────
if page == "🖼️ DR Grading (CNN)":
    st.header("Diabetic Retinopathy Grading")
    
    uploaded_file = st.file_uploader(
        "Upload a retinal fundus image", 
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True)
        
        with col2:
            st.subheader("Prediction")
            # Load model and predict
            # (model loading code here)
            
            grade_labels = {
                0: ("No DR", "🟢", "#2ecc71"),
                1: ("Mild DR", "🟡", "#f1c40f"),
                2: ("Moderate DR", "🟠", "#e67e22"),
                3: ("Severe DR", "🔴", "#e74c3c"),
                4: ("Proliferative DR", "🟣", "#8e44ad"),
            }
            
            # Display prediction result
            predicted_grade = 2  # Replace with actual model output
            label, icon, color = grade_labels[predicted_grade]
            
            st.metric("Predicted Grade", f"{icon} {label}")
            
            # Confidence bar
            st.subheader("Class Probabilities")
            probs = [0.05, 0.08, 0.65, 0.15, 0.07]  # Example
            fig = go.Figure(go.Bar(
                x=probs,
                y=[f"{v[0]}" for v in grade_labels.values()],
                orientation='h',
                marker_color=['#2ecc71','#f1c40f','#e67e22','#e74c3c','#8e44ad']
            ))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
```

### 12.2 Presentation Slide Outline

```
SLIDE 01: Title + Team
  - Project name, team members, bootcamp details
  
SLIDE 02: Problem & Clinical Motivation
  - DR prevalence statistics (IDF 2023 data)
  - Manual grading bottleneck
  - Proposed system overview (1-slide architecture)

SLIDE 03: Dataset Overview
  - APTOS 2019: 3,662 images, 5 classes
  - Class distribution bar chart + sample images per grade
  - Ben Graham preprocessing: before vs. after comparison

SLIDE 04: CNN Architecture
  - ResNet-50 backbone diagram
  - Two-phase fine-tuning strategy table
  - Training curves (loss + QWK over epochs)

SLIDE 05: Evaluation Results
  - Results comparison table (ResNet-50 vs. VGG-16 vs. Scratch)
  - Confusion matrix heatmap
  - Per-class AUC-ROC

SLIDE 06: Explainability (Grad-CAM)
  - 2×3 grid: DR grades 0–4 + one failure case
  - Clinical lesion labels (microaneurysms, hemorrhages, exudates)
  - Discussion: XAI builds clinical trust

SLIDE 07: Patient Similarity Network
  - Network construction methodology diagram
  - Gephi/PyVis visualization (color-coded by DR grade)
  - Community detection results

SLIDE 08: Network Insights
  - High-risk bridge patient analysis
  - Community DR grade distribution
  - Homophily score interpretation

SLIDE 09: Integrated System Demo
  - Streamlit dashboard screenshots / live demo

SLIDE 10: Limitations & Future Work
  - 15-day scope limitations
  - Extension: GNN (Graph Neural Network) classifier
  - Extension: Longitudinal patient tracking

SLIDE 11: Conclusion
  - Key takeaways
  - GitHub link + QR code for demo
```

---

## 13. GitHub Repository Guide

### 13.1 README Structure

```markdown
# DR Grading + Patient Risk Network
> Explainable CNN-based Diabetic Retinopathy grading with patient similarity 
> network analysis | IIIT Kottayam Data Science Bootcamp 2026

## Results Summary
| Model     | QWK    | AUC-ROC | F1-Macro |
|-----------|--------|---------|----------|
| ResNet-50 | 0.XXX  | 0.XXX   | 0.XXX    |
| VGG-16    | 0.XXX  | 0.XXX   | 0.XXX    |
| Scratch   | 0.XXX  | 0.XXX   | 0.XXX    |

## Quick Start
...setup instructions...

## Project Structure
...directory tree...

## Reproducibility
Set seed: 42 | Hardware: GPU recommended
```

### 13.2 Git Commit Convention

```
feat: Add ResNet-50 two-phase fine-tuning
fix: Correct class weight computation for imbalanced dataset
docs: Add Grad-CAM interpretation guide
eval: Add QWK vs VGG-16 baseline comparison
viz: Export patient similarity network to Gephi
```

---

## 14. Risk Register & Contingencies

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Kaggle API download failure | Medium | High | Download via browser; store in Google Drive |
| GPU unavailable (Colab free tier) | High | High | Use Colab Pro trial / reduce batch size to 16 |
| Training time exceeds 15 days | Low | High | Use timm EfficientNet-B0 (faster convergence) |
| Class 3, 4 low recall | High | Medium | Apply focal loss + oversampling |
| Network too sparse (few edges) | Medium | Medium | Lower threshold from 0.85 → 0.75 |
| Network too dense | Medium | Medium | Increase threshold + reduce max_edges_per_node |
| Streamlit deployment issues | Low | Low | Use localhost; record screen for presentation |

---

## 15. Portfolio Maximization Checklist

### Code Quality
- [ ] All notebooks have markdown explanations of every decision
- [ ] All functions have docstrings with Args/Returns
- [ ] Config variables separated into `config.py` (not hardcoded)
- [ ] Requirements.txt with pinned versions
- [ ] `.gitignore` excludes data/ and model checkpoints (>100MB)
- [ ] Results are reproducible with `seed=42` everywhere

### Portfolio Signals
- [ ] GitHub README has results table, architecture diagram, GIF demo
- [ ] Grad-CAM visualizations saved as high-resolution PNGs
- [ ] Gephi `.gexf` file committed for reviewers to explore
- [ ] Streamlit app deployed on HuggingFace Spaces (free)
- [ ] Jupyter notebooks exported as HTML for non-technical reviewers

### Research Readiness
- [ ] Ablation study: with vs. without Ben Graham preprocessing
- [ ] Ablation study: Phase 1 only vs. two-phase training
- [ ] Failure case analysis: misclassified samples Grad-CAM inspection
- [ ] Network homophily comparison: random graph vs. patient graph
- [ ] All metrics reported on held-out TEST set (not validation)

### Presentation
- [ ] 11 slides, ≤12 minutes delivery
- [ ] Clinical impact quantified (e.g., "detects X% of Severe DR cases")
- [ ] GitHub QR code on final slide
- [ ] Demo video recorded as backup for live demo failures

---

*Blueprint version 1.0 | IIIT Kottayam Data Science Bootcamp 2026*  
*Project: Explainable DR Grading + Patient Comorbidity Networks*
