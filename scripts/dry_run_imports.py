"""Dry-run import test for main.py — verifies all module imports resolve."""
import sys, io
from pathlib import Path
# Ensure project root (parent of scripts/) is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

errors = []

def try_import(label, fn):
    try:
        fn()
        print(f"  [OK] {label}")
    except Exception as e:
        print(f"  [!!] {label}  ->  {e}")
        errors.append((label, str(e)))

print("=" * 55)
print("  Praxis -- Import Dry-Run")
print("=" * 55)

try_import("src.utils.config",          lambda: __import__("src.utils.config"))
try_import("src.data (transforms)",     lambda: __import__("src.data", fromlist=["get_train_transforms"]))
try_import("src.data.dataset",          lambda: __import__("src.data.dataset", fromlist=["create_data_loaders"]))
try_import("src.models (build+Trainer)",lambda: __import__("src.models", fromlist=["build_resnet50","Trainer"]))
try_import("src.evaluation (metrics)",  lambda: __import__("src.evaluation", fromlist=["compute_all_metrics"]))
try_import("src.utils.visualization",   lambda: __import__("src.utils.visualization", fromlist=["plot_training_curves"]))
try_import("src.network.analysis",      lambda: __import__("src.network.analysis", fromlist=["NetworkAnalyzer"]))
try_import("src.explainability.gradcam (GradCAMVisualizer)",
           lambda: __import__("src.explainability.gradcam", fromlist=["GradCAMVisualizer"]))

# Config + device
from src.utils.config import load_config, get_device
cfg = load_config("configs/default.yaml")
dev = get_device()
try_import("config loads + device",     lambda: None)

print()
if not errors:
    print(f"  ALL IMPORTS OK -- device: {dev}")
    print(f"  Config keys: {list(cfg.keys())}")
else:
    print(f"  {len(errors)} import error(s):")
    for label, err in errors:
        print(f"    {label}: {err}")
print("=" * 55)
