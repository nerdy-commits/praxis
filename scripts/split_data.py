"""
Split the real APTOS 2019 train.csv into stratified 70/15/15 train/val/test splits.
Images stay in the original location — only CSVs are created.
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APTOS_ROOT = PROJECT_ROOT / "data" / "raw" / "aptos2019"
TRAIN_CSV  = APTOS_ROOT / "train.csv"
OUT_DIR    = APTOS_ROOT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

print("=" * 55)
print("  Praxis -- APTOS 2019 Data Splitter")
print("=" * 55)

# Load
df = pd.read_csv(TRAIN_CSV)
print(f"\n  Source: {TRAIN_CSV}")
print(f"  Total rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")

# Check images are accessible
img_dir = APTOS_ROOT / "train_images"
n_imgs = len(list(img_dir.glob("*.png")))
print(f"  Images found: {n_imgs}")

# Stratified 70 / 15 / 15
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["diagnosis"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["diagnosis"], random_state=SEED
)

# Save splits
train_df.to_csv(OUT_DIR / "train_split.csv", index=False)
val_df.to_csv(OUT_DIR   / "val_split.csv",   index=False)
test_df.to_csv(OUT_DIR  / "test_split.csv",  index=False)

print(f"\n  Splits saved to {OUT_DIR}/")
print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

print(f"\n  {'Grade':<22} {'Train':>6} {'Val':>6} {'Test':>6}")
print("  " + "-" * 42)
for g, name in enumerate(CLASS_NAMES):
    tr = int((train_df.diagnosis == g).sum())
    va = int((val_df.diagnosis   == g).sum())
    te = int((test_df.diagnosis  == g).sum())
    print(f"  {g} {name:<20} {tr:>6} {va:>6} {te:>6}")

print("\n  [OK] Data splits ready!")
print("  Next: python main.py")
