# =============================================================================
#  CNN Classifiers — ResNet-50 & VGG-16 for DR Grading
# =============================================================================
"""
Transfer learning classifiers for 5-class Diabetic Retinopathy grading.

Architecture strategy:
    1. Load ImageNet-pretrained backbone
    2. Freeze early layers (low-level features transfer well)
    3. Replace classifier head with DR-specific FC layers
    4. Fine-tune last N residual blocks

Why ResNet-50 over deeper variants:
    - APTOS dataset is only ~3.6K images → deeper models overfit
    - ResNet-50 is the sweet spot for this data size
    - Skip connections help with gradient flow during fine-tuning
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy classifier with configurable backbone.

    Parameters
    ----------
    backbone : nn.Module
        Feature extractor (e.g., ResNet-50 without FC layer).
    feature_dim : int
        Dimensionality of backbone output features.
    num_classes : int
        Number of DR severity grades (default: 5).
    fc_hidden : int
        Hidden layer size in the classifier head.
    dropout : float
        Dropout probability for regularization.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 5,
        fc_hidden: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without classification head."""
        with torch.no_grad():
            features = self.backbone(x)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
        return features

    def unfreeze_backbone_blocks(self, unfreeze_blocks: int):
        """Unfreeze the last N blocks of the backbone for Phase 2 training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if unfreeze_blocks > 0:
            layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
            for layer in layers[-unfreeze_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True


def _freeze_layers(model: nn.Module, unfreeze_blocks: int = 2) -> nn.Module:
    """
    Freeze all parameters, then unfreeze the last N blocks.
    For ResNet: blocks are layer1, layer2, layer3, layer4.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N layer blocks
    if unfreeze_blocks > 0:
        layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        for layer in layers[-unfreeze_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

    return model


def build_resnet50(
    num_classes: int = 5,
    fc_hidden: int = 512,
    dropout: float = 0.3,
    pretrained: bool = True,
    unfreeze_blocks: int = 2,
) -> DRClassifier:
    """
    Build a ResNet-50 based DR classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    fc_hidden : int
        Hidden layer size in classifier head.
    dropout : float
        Dropout rate.
    pretrained : bool
        Whether to use ImageNet pretrained weights.
    unfreeze_blocks : int
        Number of residual blocks to unfreeze from the end.

    Returns
    -------
    DRClassifier
        Ready-to-train model.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    resnet = models.resnet50(weights=weights)

    # Apply freezing strategy
    if pretrained and unfreeze_blocks < 4:
        resnet = _freeze_layers(resnet, unfreeze_blocks)

    # Remove original FC layer, use identity to extract features
    feature_dim = resnet.fc.in_features  # 2048
    resnet.fc = nn.Identity()

    return DRClassifier(
        backbone=resnet,
        feature_dim=feature_dim,
        num_classes=num_classes,
        fc_hidden=fc_hidden,
        dropout=dropout,
    )


def build_vgg16(
    num_classes: int = 5,
    fc_hidden: int = 512,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> DRClassifier:
    """
    Build a VGG-16 based DR classifier (baseline comparison).

    VGG-16 is used as a baseline to demonstrate ResNet-50's
    superiority via skip connections on this dataset size.
    """
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    vgg = models.vgg16(weights=weights)

    # Freeze feature extractor
    if pretrained:
        for param in vgg.features.parameters():
            param.requires_grad = False

    # Replace classifier
    feature_dim = 512 * 7 * 7  # VGG-16 feature map at 224x224 input
    vgg.classifier = nn.Identity()

    return DRClassifier(
        backbone=vgg,
        feature_dim=feature_dim,
        num_classes=num_classes,
        fc_hidden=fc_hidden,
        dropout=dropout,
    )
