# =============================================================================
#  Training Loop — DR Classifier Trainer
# =============================================================================
"""
Training engine with:
  - Mixed precision (AMP) for ~2× speedup on CUDA GPUs
  - Early stopping on validation QWK
  - Cosine / Step / Plateau LR scheduling
  - Class-weighted CrossEntropy + label smoothing
  - Best model checkpoint saving

Designed for healthcare-grade evaluation where standard accuracy
is insufficient due to class imbalance and ordinal class structure.
"""

import time
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
# GradScaler: use torch.amp on 2.2+, fall back to torch.cuda.amp on 2.1.x
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
)
from tqdm import tqdm

from ..evaluation.metrics import quadratic_weighted_kappa


class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after last improvement.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        'max' for metrics to maximize (QWK), 'min' for losses.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Training engine for the DR classifier with AMP support.

    Tracks loss, QWK, and learning rate per epoch.
    Saves best model checkpoint based on validation QWK.

    Parameters
    ----------
    model : nn.Module
        The DR classifier model.
    device : torch.device
        Target device (cuda / cpu).
    config : dict
        Training configuration from YAML.
    save_dir : str
        Directory to save model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
        save_dir: str = "outputs/models",
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Mixed-precision scaler — no-op on CPU
        cfg_train = config["training"]
        self.use_amp = cfg_train.get("mixed_precision", False) and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        if self.use_amp:
            print(f"  [AMP] Mixed precision enabled -- CUDA device: {device}")
        else:
            print(f"  [INFO] Running on {device} (AMP {'disabled (CPU)' if device.type == 'cpu' else 'disabled by config'})")

        # History tracking
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_qwk": [],
            "val_qwk": [],
            "lr": [],
        }

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config, only for trainable parameters."""
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        cfg = self.config["training"]

        if cfg["optimizer"] == "adam":
            return Adam(
                trainable,
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
            )
        elif cfg["optimizer"] == "sgd":
            return SGD(
                trainable,
                lr=cfg["learning_rate"],
                momentum=0.9,
                weight_decay=cfg["weight_decay"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    def _build_scheduler(self, optimizer):
        """Build LR scheduler from config."""
        cfg = self.config["training"]["scheduler"]
        if cfg["type"] == "cosine":
            return CosineAnnealingLR(
                optimizer, T_max=self.config["training"]["epochs"]
            )
        elif cfg["type"] == "step":
            return StepLR(
                optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"]
            )
        elif cfg["type"] == "plateau":
            return ReduceLROnPlateau(
                optimizer, mode="max", patience=cfg["patience"], factor=cfg["gamma"]
            )
        else:
            return None

    def _build_criterion(self, class_weights: Optional[torch.Tensor]) -> nn.Module:
        """CrossEntropy with optional class weighting and label smoothing."""
        cfg = self.config["training"]
        label_smoothing = cfg.get("label_smoothing", 0.0)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple:
        """Run one training epoch with AMP support. Returns (avg_loss, qwk)."""
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(loader, desc="  Train", leave=False, ncols=80):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = criterion(logits, labels)

            self.scaler.scale(loss).backward()
            # Gradient clipping — prevents exploding gradients during fine-tuning
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(loader.dataset)
        qwk = quadratic_weighted_kappa(all_labels, all_preds)
        return avg_loss, qwk

    @torch.no_grad()
    def _validate_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple:
        """Run one validation epoch. Returns (avg_loss, qwk)."""
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(loader.dataset)
        qwk = quadratic_weighted_kappa(all_labels, all_preds)
        return avg_loss, qwk

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        resume_from: Optional[str] = None,
    ) -> dict:
        """Full training loop with optional checkpoint resume.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        class_weights : torch.Tensor, optional
            Inverse-frequency weights for CrossEntropyLoss.
        resume_from : str or Path, optional
            Path to a checkpoint .pth file to resume training from.
            Loads model weights and sets best_qwk to the checkpoint's val_qwk.

        Returns
        -------
        dict
            Training history with loss and QWK per epoch.
        """
        cfg    = self.config["training"]
        epochs = cfg["epochs"]

        criterion  = self._build_criterion(class_weights)
        optimizer  = self._build_optimizer()
        scheduler  = self._build_scheduler(optimizer)

        freeze_strategy = self.config["model"].get("freeze_strategy", "none")
        unfreeze_blocks = self.config["model"].get("unfreeze_blocks", 2)
        
        # Unfreeze halfway through if two-phase training is configured
        unfreeze_epoch = (epochs // 2) + 1 if freeze_strategy == "partial" else None

        early_stopping = None
        if cfg["early_stopping"]["enabled"]:
            es_cfg = cfg["early_stopping"]
            early_stopping = EarlyStopping(
                patience=es_cfg["patience"],
                min_delta=es_cfg["min_delta"],
                mode="max",
            )

        best_qwk   = -1.0
        start_epoch = 1

        # ── Resume from checkpoint ─────────────────────────────────────────
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            best_qwk    = ckpt.get("val_qwk", -1.0)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"  [RESUME] Loaded {resume_from}")
            print(f"  [RESUME] Epoch {start_epoch - 1} | Best Val QWK so far: {best_qwk:.4f}")

        print(f"\n  {'Epoch':>5}  {'TrLoss':>8}  {'TrQWK':>7}  "
              f"{'VaLoss':>8}  {'VaQWK':>7}  {'LR':>9}  {'Time':>6}")
        print("  " + "-" * 60)

        for epoch in range(start_epoch, epochs + 1):
            if unfreeze_epoch is not None and epoch == unfreeze_epoch:
                print(f"  [INFO] Two-Phase Training: Unfreezing last {unfreeze_blocks} blocks at epoch {epoch}")
                if hasattr(self.model, "unfreeze_backbone_blocks"):
                    self.model.unfreeze_backbone_blocks(unfreeze_blocks)
                
                # Rebuild optimizer with a reduced learning rate for fine-tuning
                cfg["learning_rate"] = cfg.get("learning_rate", 0.0003) * 0.1
                optimizer = self._build_optimizer()
                scheduler = self._build_scheduler(optimizer)

            t0 = time.time()

            train_loss, train_qwk = self._train_epoch(train_loader, criterion, optimizer)
            val_loss,   val_qwk   = self._validate_epoch(val_loader, criterion)

            current_lr = optimizer.param_groups[0]["lr"]

            # LR scheduling
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_qwk)
                else:
                    scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_qwk"].append(train_qwk)
            self.history["val_qwk"].append(val_qwk)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            print(
                f"  {epoch:5d}  {train_loss:8.4f}  {train_qwk:7.4f}  "
                f"{val_loss:8.4f}  {val_qwk:7.4f}  {current_lr:9.2e}  {elapsed:5.1f}s"
                + (" [best]" if val_qwk > best_qwk else "")
            )

            # Save best model
            if val_qwk > best_qwk:
                best_qwk = val_qwk
                self._save_checkpoint(epoch, val_qwk, "best_model.pth")

            # Early stopping
            if early_stopping is not None and early_stopping(val_qwk):
                print(f"\n  [STOP] Early stopping at epoch {epoch} "
                      f"(no improvement for {early_stopping.patience} epochs)")
                break

        # Save final model regardless
        self._save_checkpoint(epoch, val_qwk, "final_model.pth")
        print(f"\n  Training complete.  Best Val QWK: {best_qwk:.4f}")
        return self.history

    def _save_checkpoint(self, epoch: int, val_qwk: float, filename: str) -> None:
        """Save model checkpoint with metadata."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_qwk": val_qwk,
                "config": self.config,
            },
            self.save_dir / filename,
        )
