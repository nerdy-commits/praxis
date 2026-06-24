import sys
import os
from pathlib import Path

# Force UTF-8 on Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Ensure src module is visible
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_device
from src.data import get_train_transforms, get_val_transforms
from src.data.dataset import create_data_loaders
from src.models import build_resnet50, Trainer
from src.evaluation import compute_all_metrics

def main():
    print("=" * 60)
    print("  Praxis -- Ablation Study")
    print("=" * 60)

    config_path = "configs/default.yaml"
    base_config = load_config(config_path)
    device = get_device()
    
    # We will train each configuration for 5 epochs to save time
    epochs = 5
    base_config["training"]["epochs"] = epochs
    
    p = base_config["paths"]
    data_cfg = base_config["data"]
    
    # The 3 Configurations
    configs = [
        {"name": "A: Baseline (No Preprocess, No Freezing)", "preprocess": False, "freeze": "none"},
        {"name": "B: + Ben Graham Preprocessing", "preprocess": True, "freeze": "none"},
        {"name": "C: + Two-Phase Training", "preprocess": True, "freeze": "partial"},
    ]
    
    all_results = {}
    
    for cfg in configs:
        print(f"\n>> Running Configuration: {cfg['name']}")
        
        # Adjust config
        current_config = load_config(config_path)
        current_config["training"]["epochs"] = epochs
        
        if "preprocessing" not in current_config:
            current_config["preprocessing"] = {"ben_graham": {}}
        current_config["preprocessing"]["ben_graham"]["enabled"] = cfg["preprocess"]
        current_config["model"]["freeze_strategy"] = cfg["freeze"]
        
        # Loaders
        train_transform = get_train_transforms(data_cfg["image_size"], current_config.get("augmentation", {}))
        val_transform   = get_val_transforms(data_cfg["image_size"])
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_csv=p["train_csv"], val_csv=p["val_csv"], test_csv=p["test_csv"],
            train_image_dir=p["train_images"], val_image_dir=p["val_images"],
            test_image_dir=p["test_images"],
            train_transform=train_transform, val_transform=val_transform,
            image_size=data_cfg["image_size"],
            batch_size=8,
            num_workers=0,
            preprocess=cfg["preprocess"]
        )
        class_weights = train_loader.dataset.get_class_weights()
        
        # Build Model
        unfreeze_blocks = 0 if cfg["freeze"] == "partial" else current_config["model"]["unfreeze_blocks"]
        model = build_resnet50(
            num_classes=data_cfg["num_classes"],
            fc_hidden=current_config["model"]["fc_hidden"],
            dropout=current_config["model"]["dropout"],
            pretrained=current_config["model"]["pretrained"],
            unfreeze_blocks=unfreeze_blocks,
        ).to(device)
        
        # Train
        save_dir = Path("outputs/models/ablation") / cfg["name"].split(":")[0]
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(model, device, current_config, save_dir=str(save_dir))
        
        history = trainer.train(train_loader, val_loader, class_weights=class_weights)
        
        # Evaluate on Test
        print("  Evaluating on test set...")
        # Load best checkpoint
        ckpt_path = save_dir / "best_model.pth"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
            
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                probs  = torch.softmax(logits, dim=1)
                preds  = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                
        metrics = compute_all_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        
        all_results[cfg["name"]] = {
            "QWK":      round(metrics.get("quadratic_weighted_kappa", 0), 4),
            "AUC-ROC":  round(metrics.get("auroc_weighted", 0), 4),
            "Accuracy": round(metrics.get("accuracy", 0), 4),
        }
        
        print(f"  Result: QWK={all_results[cfg['name']]['QWK']:.4f}, AUC={all_results[cfg['name']]['AUC-ROC']:.4f}")
        
        # Clear memory before next configuration
        del model
        del trainer
        del train_loader
        del val_loader
        del test_loader
        torch.cuda.empty_cache()
        
    # Save CSV
    out_dir = Path("presentation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    comp_df = pd.DataFrame(all_results).T
    csv_path = out_dir / "Ablation_Study.csv"
    comp_df.to_csv(csv_path)
    print(f"\n  Ablation Study saved to {csv_path}")
    print(comp_df.to_string())
    
    # Plot
    fig_path = out_dir / "Ablation_Study.png"
    ax = comp_df.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="white")
    ax.set_title("Ablation Study — Effect of Preprocessing and Two-Phase Training")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Chart saved to {fig_path}")

if __name__ == "__main__":
    main()
