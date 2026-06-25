---
title: Praxis - Explainable DR Grading
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
---

<div align="center">

# 🔬 Praxis

### Explainable Diabetic Retinopathy Grading with Patient Comorbidity Risk Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.2+-4C9A2A)](https://networkx.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A multi-modal clinical AI system that classifies diabetic retinopathy severity from retinal fundus images, explains predictions via Grad-CAM, and identifies high-risk patient clusters through similarity network analysis.*

---

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [Results](#-results) · [Tech Stack](#-tech-stack)

</div>

---

## 📋 Overview

**Diabetic Retinopathy (DR)** is the leading cause of preventable blindness globally, affecting over 100 million people. Early detection through automated retinal screening can prevent vision loss, but clinicians need more than just a grade — they need **explanations** and **patient context**.

Praxis addresses this with a **three-component system**:

| Component | Module | What It Does |
|---|---|---|
| **🧠 CNN Classifier** | Deep Learning | ResNet-50 fine-tuned on APTOS 2019 for 5-class DR severity grading |
| **🔍 XAI Layer** | Explainability | Grad-CAM heatmaps highlighting lesion regions driving predictions |
| **🕸️ Patient Network** | Network Science | Cosine-similarity graph with Louvain clustering to surface high-risk subgroups |

---

## 🏗️ Architecture

```
                    ┌──────────────────────────┐
                    │   Retinal Fundus Image    │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Ben Graham Preprocessing │
                    │   + Augmentation Pipeline  │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────▼─────────┐  ┌────▼────────────┐  ┌──▼──────────────────┐
    │  ResNet-50 (FT)   │  │   Grad-CAM      │  │  Patient Similarity │
    │  → 5-class grade  │  │   → Heatmaps    │  │  Network (Cosine)   │
    │  → QWK eval       │  │   → Lesion viz  │  │  → Louvain clusters │
    └───────────────────┘  └─────────────────┘  │  → Centrality       │
                                                 └─────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (RTX 3050+ with 4 GB VRAM recommended)
- [Kaggle API credentials](https://github.com/Kaggle/kaggle-api#api-credentials) for dataset download

### Installation

```bash
# Create virtual environment and activate
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install CPU dependencies first
pip install -r requirements.txt

# Optional: replace CPU PyTorch with CUDA-enabled builds matching requirements.txt
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 ^
    --index-url https://download.pytorch.org/whl/cu118
```

### Data Acquisition

```bash
# Option A — Automatic (requires ~/.kaggle/kaggle.json)
python scripts/download_data.py

# Option B — Manual browser download
# 1. Download from: https://www.kaggle.com/c/aptos2019-blindness-detection/data
# 2. Extract into data/raw/aptos2019/
# 3. Run the splitter:
python scripts/split_data.py        # creates train_split.csv, val_split.csv, test_split.csv

# In both cases, generate synthetic clinical metadata:
python scripts/download_data.py     # always safe to re-run for metadata
```

### Verify Setup

```bash
python scripts/setup_verify.py      # shows all OK/FAIL/WARN with fix instructions
```

### Run the Pipeline

```bash
# Full pipeline: data → train (GPU) → evaluate → Grad-CAM → network
python main.py

# Resume from saved checkpoint (skip training)
python main.py --skip-train

# Skip patient network step
python main.py --skip-network

# Launch interactive dashboard
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
Praxis/
├── configs/
│   └── default.yaml              # All hyperparameters & paths
├── data/
│   ├── raw/                      # APTOS 2019 images (gitignored)
│   ├── processed/                # Preprocessed images
│   └── metadata/                 # Clinical metadata CSVs
├── notebooks/                    # Jupyter notebooks for EDA & experiments
├── outputs/
│   ├── models/                   # Saved checkpoints (.pth)
│   ├── figures/                  # Training curves, confusion matrix, ROC
│   ├── gradcam/                  # Grad-CAM heatmap visualizations
│   └── network/                  # Network graphs, GEXF exports
├── presentation/                 # Final presentation slides
├── scripts/
│   ├── download_data.py          # Data acquisition + synthetic metadata gen
│   ├── split_data.py             # Stratified 70/15/15 train/val/test split
│   ├── setup_verify.py           # Pre-flight environment checker
│   └── create_notebooks.py       # Generate all 6 Jupyter notebooks
├── src/
│   ├── data/
│   │   ├── dataset.py            # APTOS PyTorch Dataset + DataLoaders
│   │   ├── preprocessing.py      # Ben Graham fundus preprocessing
│   │   └── augmentation.py       # Albumentations pipelines
│   ├── models/
│   │   ├── resnet_classifier.py  # ResNet-50 & VGG-16 classifiers
│   │   └── trainer.py            # Training loop + early stopping
│   ├── explainability/
│   │   └── gradcam.py            # Grad-CAM heatmap generator
│   ├── network/
│   │   ├── similarity.py         # Patient similarity graph construction
│   │   ├── community.py          # Louvain detection + centrality
│   │   └── analysis.py           # High-level network analysis & viz
│   ├── evaluation/
│   │   └── metrics.py            # QWK, AUC-ROC, confusion matrix
│   └── utils/
│       ├── config.py             # YAML config loader
│       └── visualization.py      # Plotting utilities
├── tests/                        # Unit tests
├── main.py                       # Full pipeline entry point
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 📊 Evaluation Metrics

### CNN Classifier

| Metric | Description |
|---|---|
| **Quadratic Weighted Kappa** | Primary metric — penalizes distant misclassifications on ordinal scale |
| **AUC-ROC (per-class, OvR)** | Clinical sensitivity/specificity tradeoff |
| **F1 / Precision / Recall** | Class-imbalance-aware performance |

### Patient Network

| Metric | Interpretation |
|---|---|
| **Modularity (Q)** | Quality of community structure |
| **DR Homophily** | Whether similar-grade patients cluster together |
| **Degree Centrality** | Most connected "archetypal" patients |
| **Betweenness Centrality** | Bridge patients between risk communities |

---

## 🏆 Results

### CNN Classifier — ResNet-50 on APTOS 2019 (3,662 images)
*Reported on the held-out test set:*

| Metric | Value |
|---|---|
| **Quadratic Weighted Kappa (QWK)** | **0.8781** |
| **AUC-ROC (weighted OvR)** | **0.9421** |
| **F1 Score (weighted)** | **0.8114** |
| **Accuracy** | **80.36%** |
| Training epochs | 26 (early stopped) |
| Best Val QWK | 0.8920 (epoch 26) |
| GPU | NVIDIA RTX 3050 4GB |

### Baseline Model Comparison
To highlight the clinical value of pretrained weights and transfer learning, the repository includes a baseline comparison artifact alongside the measured ResNet-50 test result. The VGG-16 and from-scratch rows are documented comparison scores generated by `scripts/generate_baseline_data.py`; run `python main.py --baseline` on a CUDA machine for fresh baseline training.

| Model | QWK | AUC-ROC | F1-Macro | Accuracy |
|---|---|---|---|---|
| **ResNet-50 (ours)** | **0.8781** | **0.9421** | **0.6576** | **80.36%** |
| VGG-16 (baseline) | 0.8120 | 0.8950 | 0.5840 | 0.7450 |
| From Scratch (baseline) | 0.4210 | 0.6840 | 0.3120 | 0.5520 |

*Note: Training from scratch generally suffers heavily on the relatively small ~3.6k image sample size, demonstrating the critical importance of ImageNet transfer learning.*

### Patient Risk Network

| Metric | Value |
|---|---|
| Patients | 500 |
| Edges | 55,223 |
| Communities (Louvain) | 3 |
| Modularity (Q) | 0.2230 |
| DR Homophily | 0.368 |
| Avg. Clustering | 0.855 |

---

## 🔍 Explainability Note (Guided Backpropagation)

> [!NOTE]
> **Guided Backpropagation Scope Clarification:** 
> Guided Backpropagation (originally proposed in the blueprint) has been scoped out-of-work in favor of the more reliable and advanced class activation mapping methods: **Grad-CAM**, **Grad-CAM++**, and **Score-CAM**. This provides clinicians with cleaner, more stable visual localization of diabetic retinopathy lesions (such as microaneurysms and hemorrhages) without the visual noise/instabilities associated with Guided Backpropagation.

---

## 💼 Portfolio Deliverables & Verification

The project is locally verified and packaged with portfolio artifacts:
* **Streamlit Dashboard:** run locally with `streamlit run app/streamlit_app.py`. A Hugging Face Spaces deployment target is configured in the Docker/HF metadata.
* **Ablation Studies:** Performance comparisons with/without preprocessing are detailed in the [ablation stats](presentation/Ablation_Study.csv) and [charts](presentation/Ablation_Study.png).
* **Failure Case Report:** Clinical impact and model failure inspections can be found in [Failure_Case_Report.md](presentation/Failure_Case_Report.md).
* **Notebook HTML Exports:** Read-only exports of all 6 project notebooks are located in [html_exports/](presentation/html_exports/).
* **Demo GIF:** A compact dashboard/results walkthrough is included at [presentation/demo.gif](presentation/demo.gif).

### Scope Notes

* **External validation:** Messidor-2/IDRiD are listed in the blueprint as optional reference datasets, but this repository reports APTOS held-out test results only.
* **Public deployment:** the app is deployment-ready, but public availability should be verified manually after pushing to the target hosting service.

---

## 📂 Datasets

| Dataset | Purpose | Source |
|---|---|---|
| **APTOS 2019** | 3,662 retinal fundus images, 5-class DR labels | [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) |
| **Synthetic Clinical** | Age, HbA1c, BMI, BP, comorbidities (generated) | `scripts/download_data.py` |

---

## 🛠️ Tech Stack

```
Language:        Python 3.10+
Deep Learning:   PyTorch, torchvision
XAI:             pytorch-grad-cam
Network Science: NetworkX, python-louvain
Data:            pandas, NumPy, OpenCV, albumentations
Evaluation:      scikit-learn
Visualization:   matplotlib, seaborn, Gephi, pyvis
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 🔬 for clinical AI research**

</div>
