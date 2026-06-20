# =============================================================================
#  Praxis — Explainable DR Grading with Patient Comorbidity Risk Networks
# =============================================================================
"""
Top-level package for the Praxis project.

Components:
    - data:            Dataset loading, preprocessing, augmentation
    - models:          CNN classifiers (ResNet-50, VGG-16, baselines)
    - explainability:  Grad-CAM / SHAP explanation generators
    - network:         Patient similarity network construction & analysis
    - evaluation:      Healthcare-grade metrics (QWK, AUC-ROC, F1)
    - utils:           Configuration, visualization, common helpers
"""

__version__ = "0.1.0"
__author__ = "Praxis Team"
