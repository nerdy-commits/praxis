# src/models/__init__.py
"""CNN classifiers for Diabetic Retinopathy grading."""

from .resnet_classifier import DRClassifier, build_resnet50, build_vgg16
from .trainer import Trainer
