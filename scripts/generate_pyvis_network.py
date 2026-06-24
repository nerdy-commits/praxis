# =============================================================================
#  Generate Interactive PyVis Patient Network (Fast Bypass)
# =============================================================================
"""
Constructs the patient similarity network directly using the clinical metadata 
and true DR grades (bypassing expensive CPU ResNet-50 inference).
"""

import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root directory to python path for internal imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Force UTF-8 encoding on Windows to prevent Unicode errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.utils.config import load_config
from src.network.analysis import NetworkAnalyzer

def main():
    print(">> Generating interactive PyVis patient network (Fast Bypass)...")
    
    # 1. Load config
    config_path = "configs/default.yaml"
    config = load_config(config_path)
    paths = config["paths"]
    net_cfg = config["network"]
    
    # 2. Load Clinical Metadata
    metadata_path = Path(paths["metadata_csv"])
    if not metadata_path.exists():
        print(f"  [ERROR] Clinical metadata CSV not found at: {metadata_path}")
        return
        
    clinical_df = pd.read_csv(metadata_path)
    # Subset to first 500 patients to match the README results and avoid CPU/NetworkX bottlenecks
    clinical_df = clinical_df.head(500)
    feature_cols = net_cfg["clinical_features"]
    
    # Use existing dr_grade from metadata (clinically accurate true labels)
    dr_grades = clinical_df["dr_grade"].values if "dr_grade" in clinical_df.columns else None
    
    # 3. Build similarity network
    print("  Constructing patient similarity network and running modularity analysis...")
    analyzer = NetworkAnalyzer(
        clinical_df=clinical_df,
        feature_columns=[c for c in feature_cols if c in clinical_df.columns],
        threshold=net_cfg["edge_threshold"],
        dr_grades=dr_grades,
    )
    
    summary = analyzer.summary()
    print("\n  Network Summary statistics:")
    for k, v in summary.items():
        if v is not None:
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
            
    # Export interactive HTML
    net_dir = Path(paths["network_dir"])
    net_dir.mkdir(parents=True, exist_ok=True)
    html_path = net_dir / "patient_network.html"
    
    print(f"  Exporting pruned PyVis graph to: {html_path}")
    analyzer.export_pyvis(str(html_path))
    
    print("\n>> Interactive PyVis Patient Network Generated Successfully!")

if __name__ == "__main__":
    main()
