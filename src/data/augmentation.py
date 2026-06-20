# =============================================================================
#  Data Augmentation — Albumentations Pipelines
# =============================================================================
"""
Augmentation transforms for retinal fundus images using albumentations.

Design decisions:
    - Horizontal flip: YES (retinal images are symmetric across the vertical axis)
    - Vertical flip: NO (unnatural orientation, could confuse the model)
    - Rotation: ±30° (accounts for camera angle variation)
    - Color jitter: moderate (simulates different camera/lighting conditions)
    - No random cropping: we need the full retinal field for Grad-CAM
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 224, aug_cfg: dict = None) -> A.Compose:
    """
    Training augmentation pipeline.

    Applies spatial and color augmentations to increase training
    data diversity and reduce overfitting.
    """
    aug_cfg = aug_cfg or {}
    h_flip = 0.5 if aug_cfg.get("horizontal_flip", True) else 0.0
    v_flip = 0.5 if aug_cfg.get("vertical_flip", False) else 0.0
    rot_limit = aug_cfg.get("rotation_limit", 30)
    color = aug_cfg.get("color_jitter", {})
    
    transforms = [
        A.Resize(image_size, image_size)
    ]
    
    if h_flip > 0: transforms.append(A.HorizontalFlip(p=h_flip))
    if v_flip > 0: transforms.append(A.VerticalFlip(p=v_flip))
    
    transforms.extend([
        A.Rotate(limit=rot_limit, border_mode=0, p=0.5),
        A.ColorJitter(
            brightness=color.get("brightness", 0.2),
            contrast=color.get("contrast", 0.2),
            saturation=color.get("saturation", 0.2),
            hue=color.get("hue", 0.1),
            p=0.5,
        ),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    return A.Compose(transforms)


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Validation/test transform pipeline.

    Only resizing and normalization — no augmentation.
    Must match the normalization used in training.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
