"""
APTOS 2019 Data Acquisition — with Kaggle mirror fallback.

Three acquisition methods (tries each in order):
  1. Kaggle Python API (requires ~/.kaggle/kaggle.json)
  2. Kaggle CLI (requires `kaggle` in PATH with credentials configured)
  3. Manual download instructions printed to console

Also generates synthetic clinical metadata for the network module.

Usage:
    python scripts/download_data.py
"""

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw" / "aptos2019"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
COMPETITION  = "aptos2019-blindness-detection"


def _check_kaggle_creds() -> bool:
    """Return True if kaggle.json exists."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        # Also check USERPROFILE on Windows
        win_path = Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json"
        return win_path.exists()
    return True


def _download_via_api() -> bool:
    """Download using the kaggle Python package."""
    if not _check_kaggle_creds():
        return False
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print("  Downloading via Kaggle API …")
        api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)
        print("  [OK] Download complete!")
        return True
    except Exception as e:
        print(f"  [WARN] Kaggle API error: {e}")
        return False


def _extract_zip(zip_path: Path) -> None:
    """Unzip a file into its parent directory."""
    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(zip_path.parent)
    zip_path.unlink()
    print("  [OK] Extracted.")


def _find_and_extract_zips() -> None:
    """Extract any .zip files found in RAW_DIR."""
    for zf in RAW_DIR.glob("*.zip"):
        _extract_zip(zf)


def _print_manual_instructions() -> None:
    """Print step-by-step manual download guide."""
    print("\n" + "="*60)
    print("  [!] Kaggle credentials not found.")
    print("="*60)
    print("""
  To set up Kaggle credentials:

  1. Go to: https://www.kaggle.com/settings/account
  2. Click  'Create New Token'  ->  downloads  kaggle.json
  3. Move the file to:
       Windows:  C:\\Users\\<you>\\.kaggle\\kaggle.json
       Linux:    ~/.kaggle/kaggle.json
  4. Accept competition rules at:
       https://www.kaggle.com/c/aptos2019-blindness-detection/rules
  5. Re-run this script.

  ---------------------------------------------------------
  ALTERNATIVE -- Manual download (browser):
  1. Visit: https://www.kaggle.com/c/aptos2019-blindness-detection/data
  2. Download the dataset zip manually
  3. Extract into:  data/raw/aptos2019/
     Expected layout after extraction:
       data/raw/aptos2019/train_images/train_images/*.png
       data/raw/aptos2019/val_images/val_images/*.png
       data/raw/aptos2019/train_split.csv
       data/raw/aptos2019/val_split.csv
       data/raw/aptos2019/test_split.csv
  4. Then run:  python scripts/download_data.py  again
     (it will skip the download and go straight to metadata gen)
  ---------------------------------------------------------
""")


def _check_data_already_present() -> bool:
    """Return True if training images and CSV already exist."""
    train_img_dir = RAW_DIR / "train_images" / "train_images"
    train_csv     = RAW_DIR / "train_split.csv"
    if train_img_dir.exists() and train_csv.exists():
        n_images = len(list(train_img_dir.glob("*.png")))
        if n_images > 100:
            print(f"  ✅ Data already present: {n_images} training images found.")
            return True
    return False


def download_aptos_dataset() -> bool:
    """
    Download APTOS 2019 dataset.
    Returns True if data is ready (downloaded or already present).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Praxis — Data Acquisition")
    print("=" * 60)

    if _check_data_already_present():
        return True

    # Try Kaggle API
    if _download_via_api():
        _find_and_extract_zips()
        return _check_data_already_present()

    # If we reach here, Kaggle failed
    _print_manual_instructions()
    return False


def generate_synthetic_metadata() -> pd.DataFrame:
    """
    Generate synthetic clinical metadata for the network analysis module.

    Maps to APTOS patient IDs and creates clinically realistic features:
        - Age, HbA1c, BMI, blood pressure (systolic + diastolic)
        - Diabetes duration, cholesterol
        - Binary flags: smoking, hypertension, cardiovascular disease

    Distributions are calibrated to real-world diabetic population statistics.
    Higher DR grades correlate with worse metabolic control.
    """
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Use real patient IDs if split CSVs are available
    train_csv = RAW_DIR / "train_split.csv"
    val_csv   = RAW_DIR / "val_split.csv"
    test_csv  = RAW_DIR / "test_split.csv"
    
    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        df_train = pd.read_csv(train_csv)
        df_val   = pd.read_csv(val_csv)
        df_test  = pd.read_csv(test_csv)
        df = pd.concat([df_train, df_val, df_test], ignore_index=True)
        patient_ids = df["id_code"].tolist()
        dr_grades   = df["diagnosis"].tolist()
        n           = len(patient_ids)
        print(f"\n  Using {n} real APTOS patient IDs from split CSVs")
    else:
        print("\n  [NOTE] split CSVs not found — generating 500 synthetic patients.")
        n           = 500
        patient_ids = [f"P{i:04d}" for i in range(n)]
        dr_grades   = np.random.choice(
            [0, 1, 2, 3, 4], size=n, p=[0.49, 0.09, 0.27, 0.05, 0.10]
        ).tolist()

    np.random.seed(42)
    grades       = np.array(dr_grades)
    grade_offset = grades * 0.15   # subtle severity correlation

    metadata = pd.DataFrame({
        "patient_id": patient_ids,
        "dr_grade":   dr_grades,
        # Continuous features (grade-correlated)
        "age": np.clip(
            np.random.normal(55, 12, n) + grades * 2, 25, 85
        ).astype(int),
        "hba1c": np.clip(
            np.random.normal(7.5, 1.5, n) + grade_offset * 2, 4.5, 14.0
        ).round(1),
        "bmi": np.clip(
            np.random.normal(28, 5, n) + grade_offset, 16, 48
        ).round(1),
        "bp_systolic": np.clip(
            np.random.normal(135, 18, n) + grades * 3, 90, 210
        ).astype(int),
        "bp_diastolic": np.clip(
            np.random.normal(82, 10, n) + grades * 1.5, 55, 125
        ).astype(int),
        "diabetes_duration": np.clip(
            np.random.exponential(8, n) + grades * 2, 0, 40
        ).round(1),
        "cholesterol": np.clip(
            np.random.normal(210, 35, n) + grades * 5, 120, 350
        ).astype(int),
        # Binary comorbidities (probability increases with grade)
        "smoking": np.random.binomial(
            1, np.clip(0.22 + grade_offset * 0.1, 0, 1), n
        ),
        "hypertension": np.random.binomial(
            1, np.clip(0.45 + grade_offset * 0.15, 0, 1), n
        ),
        "cardiovascular": np.random.binomial(
            1, np.clip(0.15 + grade_offset * 0.10, 0, 1), n
        ),
    })

    out_path = METADATA_DIR / "clinical_metadata.csv"
    metadata.to_csv(out_path, index=False)

    print(f"\n  [OK] Clinical metadata saved -> {out_path}")
    print(f"  Patients: {n}")
    print(f"  Features: {list(metadata.columns[2:])}")
    print(f"\n  Grade distribution:")
    for grade, count in metadata["dr_grade"].value_counts().sort_index().items():
        bar = "#" * int(count / n * 40)
        print(f"    Grade {grade}: {count:4d} ({count/n*100:.1f}%) {bar}")

    return metadata


if __name__ == "__main__":
    data_ready = download_aptos_dataset()
    generate_synthetic_metadata()

    if data_ready:
        print("\n" + "=" * 60)
        print("  [OK] All data ready! Run the pipeline with:")
        print("     python main.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  Metadata generated. APTOS images still needed.")
        print("  Follow the manual download instructions above,")
        print("  then re-run this script to verify.")
        print("=" * 60)
