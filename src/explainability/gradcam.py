# =============================================================================
#  Grad-CAM Explainability Module
# =============================================================================
"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for DR classification.

Generates heatmap overlays on retinal fundus images showing which regions
the CNN attends to when making DR severity predictions.

Clinical interpretation:
    - Bright red regions → high activation → likely lesion areas
    - Microaneurysms, hemorrhages, exudates should activate strongly
    - Failure cases where the model attends to irrelevant regions
      indicate potential misclassification risk

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
           Networks via Gradient-based Localization" (ICCV 2017)
"""

from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMExplainer:
    """
    Generate Grad-CAM heatmap explanations for DR predictions.

    Parameters
    ----------
    model : nn.Module
        Trained DR classifier.
    target_layer : nn.Module
        The convolutional layer to compute Grad-CAM on.
        For ResNet-50: model.backbone.layer4[-1]
    device : torch.device
        Computation device.
    method : str
        CAM method: 'gradcam', 'gradcam++', or 'scorecam'.
    """

    METHOD_MAP = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
    }

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
        method: str = "gradcam",
    ):
        self.model = model.eval()
        self.device = device

        cam_class = self.METHOD_MAP.get(method)
        if cam_class is None:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {list(self.METHOD_MAP)}"
            )

        self.cam = cam_class(model=model, target_layers=[target_layer])

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Preprocessed image tensor of shape (1, 3, H, W) or (3, H, W).
        target_class : int, optional
            Class to generate heatmap for. If None, uses predicted class.

        Returns
        -------
        np.ndarray
            Grayscale heatmap of shape (H, W) with values in [0, 1].
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]

        grayscale_cam = self.cam(input_tensor=image_tensor, targets=targets)
        return grayscale_cam[0]  # shape: (H, W)

    def overlay_heatmap(
        self,
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original RGB image.

        Parameters
        ----------
        image_rgb : np.ndarray
            Original image in RGB, shape (H, W, 3), values in [0, 1].
        heatmap : np.ndarray
            Grad-CAM heatmap, shape (H, W), values in [0, 1].
        alpha : float
            Heatmap transparency (0 = invisible, 1 = opaque).

        Returns
        -------
        np.ndarray
            Overlaid image in RGB, values in [0, 1].
        """
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb.astype(np.float32) / 255.0

        visualization = show_cam_on_image(
            image_rgb, heatmap, use_rgb=True, image_weight=1 - alpha
        )
        return visualization

    def explain_batch(
        self,
        images: List[torch.Tensor],
        original_images: List[np.ndarray],
        labels: List[int],
        predictions: List[int],
        save_dir: str,
        alpha: float = 0.5,
    ) -> None:
        """
        Generate and save Grad-CAM visualizations for a batch of images.

        Creates a figure with three columns per image:
            1. Original image
            2. Grad-CAM heatmap
            3. Overlay

        Parameters
        ----------
        images : list of torch.Tensor
            Preprocessed image tensors.
        original_images : list of np.ndarray
            Original RGB images (for overlay).
        labels : list of int
            Ground truth labels.
        predictions : list of int
            Model predictions.
        save_dir : str
            Directory to save visualization PNGs.
        alpha : float
            Heatmap overlay transparency.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

        for i, (img_tensor, orig_img, label, pred) in enumerate(
            zip(images, original_images, labels, predictions)
        ):
            heatmap = self.generate_heatmap(img_tensor, target_class=pred)

            # Ensure original image is float [0, 1]
            if orig_img.max() > 1.0:
                orig_img = orig_img.astype(np.float32) / 255.0

            # Resize heatmap to match original image
            h, w = orig_img.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            overlay = self.overlay_heatmap(orig_img, heatmap_resized, alpha)

            # Create 3-panel figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            status = "✓" if label == pred else "✗"

            axes[0].imshow(orig_img)
            axes[0].set_title("Original", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(heatmap_resized, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay", fontsize=12)
            axes[2].axis("off")

            correct = "OK" if label == pred else "XX"
            fig.suptitle(
                f"[{correct}] True: {class_names[label]}  Pred: {class_names[pred]}",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(save_path / f"gradcam_{i:04d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


class GradCAMVisualizer:
    """
    High-level Grad-CAM interface used by main.py.

    Wraps GradCAMExplainer with automatic target-layer detection for
    ResNet-50 (layer4[-1]) and a simple batch_generate() entry point.

    Parameters
    ----------
    model : nn.Module
        Trained DR classifier (DRClassifier wrapping a ResNet-50 backbone).
    device : torch.device
        Computation device.
    method : str
        CAM method: 'gradcam' | 'gradcam++' | 'scorecam'.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        method: str = "gradcam",
    ):
        self.model  = model.eval().to(device)
        self.device = device

        # Auto-detect target layer: ResNet-50 backbone -> layer4[-1]
        target_layer = self._get_target_layer(model)
        self.explainer = GradCAMExplainer(
            model=model,
            target_layer=target_layer,
            device=device,
            method=method,
        )

    @staticmethod
    def _get_target_layer(model: nn.Module) -> nn.Module:
        """Return the last conv block of the backbone."""
        # DRClassifier stores the backbone as model.backbone
        backbone = getattr(model, "backbone", model)
        if hasattr(backbone, "layer4"):
            return backbone.layer4[-1]
        # Fallback: walk children and return the last Conv2d-containing module
        last = None
        for m in backbone.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise ValueError("Could not auto-detect target layer for Grad-CAM.")
        return last

    def batch_generate(
        self,
        data_loader,
        n_samples: int = 25,
        output_dir: str = "outputs/gradcam",
    ) -> None:
        """
        Generate Grad-CAM heatmaps for n_samples images from data_loader.

        Samples are class-balanced: n_samples // 5 images per DR grade.
        Output filenames: {GradeName}_pred{pred}_correct|wrong_{idx:04d}.png

        Parameters
        ----------
        data_loader : DataLoader
            Test set loader (images must be normalised tensors).
        n_samples : int
            Total number of images to process (split across 5 grades).
        output_dir : str
            Directory to write PNG files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
        per_class   = max(1, n_samples // 5)

        # ImageNet denormalisation constants
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        # Collect class-balanced samples
        buckets = {i: [] for i in range(5)}

        self.model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                images_dev = images.to(self.device)
                logits = self.model(images_dev)
                preds  = logits.argmax(dim=1).cpu().numpy()

                for img_t, label, pred in zip(images, labels.numpy(), preds):
                    cls = int(label)
                    if len(buckets[cls]) < per_class:
                        orig = img_t.permute(1, 2, 0).numpy()
                        orig = (orig * std + mean).clip(0, 1).astype(np.float32)
                        buckets[cls].append((img_t, orig, cls, int(pred)))

                if all(len(v) >= per_class for v in buckets.values()):
                    break

        # Flatten + generate
        all_samples = []
        for cls_samples in buckets.values():
            all_samples.extend(cls_samples)

        saved = 0
        for img_t, orig, label, pred in all_samples:
            heatmap = self.explainer.generate_heatmap(img_t, target_class=pred)
            h, w = orig.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            overlay = self.explainer.overlay_heatmap(orig, heatmap_resized, alpha=0.5)

            correct_str = "correct" if label == pred else "wrong"
            grade_name  = CLASS_NAMES[label]
            filename    = f"{grade_name}_pred{pred}_{correct_str}_{saved:04d}.png"

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            correct_label = "CORRECT" if label == pred else "WRONG"
            axes[0].imshow(orig)
            axes[0].set_title("Original", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(heatmap_resized, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay", fontsize=12)
            axes[2].axis("off")

            class_names_full = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
            fig.suptitle(
                f"[{correct_label}]  True: {class_names_full[label]}  "
                f"Pred: {class_names_full[pred]}",
                fontsize=14, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(out_path / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved += 1

        print(f"  Grad-CAM: {saved} heatmaps saved -> {output_dir}")
