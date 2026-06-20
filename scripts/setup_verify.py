# =============================================================================
#  Setup Verifier — Run before python main.py
# =============================================================================
"""
Checks that the environment is ready to run the full Praxis pipeline.

Usage:
    python scripts/setup_verify.py

Exit code 0 = all good, exit code 1 = issues found.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "[OK]"
FAIL = "[!!]"
WARN = "[??]"

issues = []
warnings = []


def check(label, condition, fix="", warn_only=False):
    if condition:
        print(f"  {PASS} {label}")
    elif warn_only:
        print(f"  {WARN} {label}")
        warnings.append(f"{label}  ->  {fix}")
    else:
        print(f"  {FAIL} {label}")
        issues.append(f"{label}  ->  {fix}")


print("=" * 60)
print("  Praxis -- Environment Setup Verifier")
print("=" * 60)

# ── Python version ────────────────────────────────────────────────────────────
print("\n[Python]")
import sys as _sys
v = _sys.version_info
check(f"Python >= 3.9  (found {v.major}.{v.minor}.{v.micro})", v >= (3, 9))

# ── PyTorch + CUDA ────────────────────────────────────────────────────────────
print("\n[PyTorch / CUDA]")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    check(f"PyTorch installed  (v{torch.__version__})", True)
    check(
        f"CUDA available  (GPU: {torch.cuda.get_device_name(0) if cuda_ok else 'none'})",
        cuda_ok,
        fix="pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 "
            "--index-url https://download.pytorch.org/whl/cu124",
        warn_only=True,
    )
    if cuda_ok:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        check(f"VRAM >= 4 GB  (found {vram_gb:.1f} GB)", vram_gb >= 4,
              fix="Use batch_size=8 and mixed_precision=true in configs/default.yaml",
              warn_only=True)
except ImportError:
    check("PyTorch installed", False, fix="pip install torch torchvision")

# ── Key packages ──────────────────────────────────────────────────────────────
print("\n[Packages]")
packages = [
    ("numpy",           "numpy"),
    ("pandas",          "pandas"),
    ("cv2",             "opencv-python"),
    ("sklearn",         "scikit-learn"),
    ("albumentations",  "albumentations"),
    ("networkx",        "networkx"),
    ("community",       "python-louvain"),
    ("matplotlib",      "matplotlib"),
    ("seaborn",         "seaborn"),
    ("tqdm",            "tqdm"),
    ("yaml",            "pyyaml"),
    ("streamlit",       "streamlit"),
    ("plotly",          "plotly"),
    ("pyvis",           "pyvis"),
    ("timm",            "timm"),
]
for mod, pkg in packages:
    try:
        __import__(mod)
        check(f"{pkg}", True)
    except ImportError:
        check(f"{pkg}", False, fix=f"pip install {pkg}")

# ── Data files ────────────────────────────────────────────────────────────────
print("\n[Data]")
from src.utils.config import load_config
cfg_paths = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))["paths"]

aptos_root = Path("E:/New folder/aptos2019-blindness-detection")
img_dir    = aptos_root / "train_images"
train_csv  = PROJECT_ROOT / cfg_paths["train_csv"]
val_csv    = PROJECT_ROOT / cfg_paths["val_csv"]
test_csv   = PROJECT_ROOT / cfg_paths["test_csv"]
meta_csv   = PROJECT_ROOT / cfg_paths["metadata_csv"]

n_imgs = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
check(f"APTOS images  ({n_imgs} found, need >= 3000)",
      n_imgs >= 3000,
      fix="Dataset should be at: E:/New folder/aptos2019-blindness-detection/train_images/")
check("train_split.csv", train_csv.exists(),
      fix="Run: python scripts/split_data.py")
check("val_split.csv",   val_csv.exists(),
      fix="Run: python scripts/split_data.py")
check("test_split.csv",  test_csv.exists(),
      fix="Run: python scripts/split_data.py")
check("clinical_metadata.csv", meta_csv.exists(),
      fix="Run: python scripts/download_data.py")

if train_csv.exists():
    import pandas as pd
    df = pd.read_csv(train_csv)
    check(f"train CSV has 'id_code' + 'diagnosis' columns",
          "id_code" in df.columns and "diagnosis" in df.columns)
    check(f"train split has >= 2000 rows (found {len(df)})", len(df) >= 2000)

# ── Outputs ───────────────────────────────────────────────────────────────────
print("\n[Outputs]")
for d in ["outputs/models", "outputs/figures", "outputs/gradcam", "outputs/network", "outputs/results"]:
    p = PROJECT_ROOT / d
    p.mkdir(parents=True, exist_ok=True)
    check(f"{d}/  (created if missing)", True)

# ── Kaggle ────────────────────────────────────────────────────────────────────
print("\n[Kaggle API]")
kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
check(
    "~/.kaggle/kaggle.json present",
    kaggle_json.exists(),
    fix="Download from: https://www.kaggle.com/settings -> 'Create New Token'\n"
        "         Move to C:\\Users\\<you>\\.kaggle\\kaggle.json",
    warn_only=True,
)

# ── Config ────────────────────────────────────────────────────────────────────
print("\n[Config]")
try:
    from src.utils.config import load_config
    cfg = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
    check("configs/default.yaml loads OK", True)
    bs = cfg["training"]["batch_size"]
    mp = cfg["training"].get("mixed_precision", False)
    check(f"batch_size={bs}  (recommended 16 for 4GB GPU)", bs <= 32)
    check(f"mixed_precision={mp}  (recommended: true for GPU)", mp, warn_only=True)
except Exception as e:
    check(f"configs/default.yaml  ({e})", False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if not issues and not warnings:
    print("  ALL CHECKS PASSED -- ready to run:  python main.py")
elif not issues:
    print(f"  READY (with {len(warnings)} optional warning(s)):")
    for w in warnings:
        print(f"    - {w}")
    print("\n  Run:  python main.py")
else:
    print(f"  {len(issues)} BLOCKING ISSUE(S) found:")
    for i, issue in enumerate(issues, 1):
        print(f"\n  {i}. {issue}")
    if warnings:
        print(f"\n  {len(warnings)} optional warning(s):")
        for w in warnings:
            print(f"    - {w}")
print("=" * 60)

sys.exit(1 if issues else 0)
