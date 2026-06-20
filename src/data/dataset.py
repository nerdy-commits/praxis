# =============================================================================
#  APTOS 2019 Dataset — PyTorch Dataset Implementation
# =============================================================================
"""
Custom PyTorch Dataset for the APTOS 2019 Blindness Detection challenge.

Handles:
    - Loading retinal fundus images from disk
    - Applying Ben Graham preprocessing
    - Train/val/test split management
    - On-the-fly augmentation via albumentations
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .preprocessing import preprocess_image


class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 retinal fundus images.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns ['id_code', 'diagnosis'].
    image_dir : str
        Directory containing the .png fundus images.
    transform : callable, optional
        Albumentations transform pipeline to apply.
    preprocess : bool
        Whether to apply Ben Graham preprocessing.
    image_size : int
        Target image size (square).
    """

    CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[Callable] = None,
        preprocess: bool = True,
        image_size: int = 224,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.preprocess = preprocess
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_name = f"{row['id_code']}.png"
        img_path = self.image_dir / img_name

        # --- Memory-safe image loading -----------------------------------
        # PIL.Image.thumbnail() resizes BEFORE fully decoding the pixel
        # buffer, so we never allocate the full ~25MB for a 3000x3000 image.
        # cv2.imread allocates the entire uncompressed buffer upfront and
        # crashes with cv2.error: Insufficient memory on 4GB-VRAM machines.
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(str(img_path)).convert("RGB")
            MAX_LOAD_SIZE = 512
            if max(pil_img.size) > MAX_LOAD_SIZE:
                pil_img.thumbnail((MAX_LOAD_SIZE, MAX_LOAD_SIZE), PILImage.BILINEAR)
            image = np.array(pil_img)          # shape: (H, W, 3), uint8, RGB
        except Exception:
            # Fallback: cv2 with immediate resize on read error
            image = cv2.imread(str(img_path))
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            MAX_LOAD_SIZE = 512
            h, w = image.shape[:2]
            if max(h, w) > MAX_LOAD_SIZE:
                scale = MAX_LOAD_SIZE / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)

        # Apply Ben Graham preprocessing (resizes to self.image_size)
        if self.preprocess:
            image = preprocess_image(image, self.image_size)
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))

        # Apply augmentation transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert to tensor: HWC -> CHW, normalize to [0, 1]
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label = int(row["diagnosis"])
        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return a dict mapping class name → count."""
        counts = self.df["diagnosis"].value_counts().sort_index()
        return {self.CLASS_NAMES[i]: int(counts.get(i, 0)) for i in range(5)}

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced training."""
        counts = self.df["diagnosis"].value_counts().sort_index().values
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(counts)
        return torch.tensor(weights, dtype=torch.float32)


def create_data_loaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    train_image_dir: str,
    val_image_dir: str,
    test_image_dir: str,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    preprocess: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from pre-split CSV files.

    The APTOS dataset mirror provides separate CSVs and image directories
    for each split, so no manual splitting is needed.

    Parameters
    ----------
    train_csv, val_csv, test_csv : str
        Paths to CSV files with ['id_code', 'diagnosis'] columns.
    train_image_dir, val_image_dir, test_image_dir : str
        Directories containing .png images for each split.
    train_transform : callable, optional
        Augmentation pipeline for training images.
    val_transform : callable, optional
        Transform pipeline for val/test images (normalize only).
    image_size : int
        Target image size (square).
    batch_size : int
        Batch size for DataLoaders.
    num_workers : int
        Number of parallel data loading workers.
    preprocess : bool
        Whether to apply Ben Graham preprocessing.

    Returns
    -------
    tuple of (train_loader, val_loader, test_loader)
    """
    train_ds = APTOSDataset(
        csv_path=train_csv,
        image_dir=train_image_dir,
        transform=train_transform,
        preprocess=preprocess,
        image_size=image_size,
    )
    val_ds = APTOSDataset(
        csv_path=val_csv,
        image_dir=val_image_dir,
        transform=val_transform,
        preprocess=preprocess,
        image_size=image_size,
    )
    test_ds = APTOSDataset(
        csv_path=test_csv,
        image_dir=test_image_dir,
        transform=val_transform,
        preprocess=preprocess,
        image_size=image_size,
    )

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader
