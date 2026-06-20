# =============================================================================
#  Unit Tests — Praxis
# =============================================================================
"""
Basic sanity tests for core modules.
Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Model Tests ───────────────────────────────────────────────────────────────
class TestResNet50:
    def test_build_resnet50_output_shape(self):
        """ResNet-50 should output [batch, 5] logits."""
        from src.models import build_resnet50
        model = build_resnet50(num_classes=5, fc_hidden=512, dropout=0.3, pretrained=False)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"

    def test_build_vgg16_output_shape(self):
        """VGG-16 baseline should also output [batch, 5]."""
        from src.models import build_vgg16
        model = build_vgg16(num_classes=5, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"

    def test_freeze_reduces_trainable_params(self):
        """Unfreezing more blocks should give more trainable parameters."""
        from src.models.resnet_classifier import _freeze_layers
        from torchvision import models
        # Two ResNets — freeze all then unfreeze 1 vs 3 blocks
        r1 = models.resnet50(weights=None)
        r2 = models.resnet50(weights=None)
        _freeze_layers(r1, unfreeze_blocks=1)
        _freeze_layers(r2, unfreeze_blocks=3)
        t1 = sum(p.numel() for p in r1.parameters() if p.requires_grad)
        t2 = sum(p.numel() for p in r2.parameters() if p.requires_grad)
        assert t1 < t2, f"Expected 1-block ({t1}) < 3-block ({t2})"


# ── Evaluation Tests ──────────────────────────────────────────────────────────
class TestMetrics:
    def test_qwk_perfect(self):
        """QWK should be 1.0 for perfect predictions."""
        from src.evaluation.metrics import quadratic_weighted_kappa
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)

    def test_qwk_range(self):
        """QWK should be in [-1, 1]."""
        from src.evaluation.metrics import quadratic_weighted_kappa
        y_true = np.random.randint(0, 5, 100)
        y_pred = np.random.randint(0, 5, 100)
        qwk = quadratic_weighted_kappa(y_true, y_pred)
        assert -1.0 <= qwk <= 1.0

    def test_compute_all_metrics_keys(self):
        """compute_all_metrics should return all expected keys."""
        from src.evaluation import compute_all_metrics
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 1, 0, 2, 4, 3])
        y_prob = np.eye(5)[y_pred]  # One-hot as probabilities
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        for key in ["quadratic_weighted_kappa", "auroc", "f1_weighted", "accuracy"]:
            assert key in metrics, f"Missing metric: {key}"


# ── Preprocessing Tests ───────────────────────────────────────────────────────
class TestPreprocessing:
    def test_ben_graham_output_shape(self):
        """Ben Graham preprocessor should return (224, 224, 3) uint8."""
        from src.data.preprocessing import BenGrahamPreprocessor
        preprocessor = BenGrahamPreprocessor(image_size=224, sigma=10)
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out = preprocessor(img)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.uint8

    def test_transforms_return_tensor(self):
        """get_val_transforms should return a float32 tensor."""
        from src.data.augmentation import get_val_transforms
        transform = get_val_transforms(224)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = transform(image=img)["image"]
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        assert out.shape == (3, 224, 224)


# ── Network Tests ─────────────────────────────────────────────────────────────
class TestPatientNetwork:
    @pytest.fixture
    def small_df(self):
        import pandas as pd
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            "age":              np.random.normal(55, 10, n),
            "hba1c":            np.random.normal(7.5, 1.5, n),
            "bmi":              np.random.normal(28, 5, n),
            "bp_systolic":      np.random.normal(135, 18, n),
            "diabetes_duration":np.random.exponential(8, n),
        })

    def test_graph_has_nodes(self, small_df):
        from src.network.similarity import build_patient_network
        G = build_patient_network(small_df, list(small_df.columns), threshold=0.70)
        assert G.number_of_nodes() == 50

    def test_graph_density_reasonable(self, small_df):
        import networkx as nx
        from src.network.similarity import build_patient_network
        G = build_patient_network(small_df, list(small_df.columns), threshold=0.80)
        density = nx.density(G)
        assert 0.0 <= density <= 1.0

    def test_community_detection(self, small_df):
        from src.network.similarity import build_patient_network
        from src.network.community import detect_communities
        G = build_patient_network(small_df, list(small_df.columns), threshold=0.75)
        partition, modularity = detect_communities(G)
        assert len(partition) == G.number_of_nodes()
        assert -1.0 <= modularity <= 1.0


# ── Config Tests ──────────────────────────────────────────────────────────────
class TestConfig:
    def test_load_config(self):
        """Config loader should return a dict with expected top-level keys."""
        from src.utils.config import load_config
        cfg = load_config("configs/default.yaml")
        for key in ["paths", "data", "model", "training", "explainability", "network"]:
            assert key in cfg, f"Missing config section: {key}"

    def test_get_device_returns_device(self):
        """get_device should return a torch.device."""
        from src.utils.config import get_device
        device = get_device()
        assert isinstance(device, torch.device)
