# =============================================================================
#  Configuration Loader
# =============================================================================
"""Load and validate YAML configuration files."""

from pathlib import Path
from typing import Dict, Any

import yaml


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Ensure output directories exist
    for dir_key in ["model_save_dir", "figures_dir", "gradcam_dir", "network_dir"]:
        dir_path = config.get("paths", {}).get(dir_key)
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    return config


def get_device():
    """Get the best available torch device."""
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
