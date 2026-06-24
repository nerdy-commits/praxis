import sys
import os
from pathlib import Path
import cv2

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_device
from src.data import get_val_transforms
from src.data.dataset import create_data_loaders
from src.models import build_resnet50

def main():
    print("=" * 60)
    print("  Praxis -- Failure Case Report Generator")
    print("=" * 60)

    config = load_config("configs/default.yaml")
    device = torch.device("cpu")
    p = config["paths"]
    
    val_transform = get_val_transforms(config["data"]["image_size"])
    
    _, _, test_loader = create_data_loaders(
        train_csv=p["train_csv"], val_csv=p["val_csv"], test_csv=p["test_csv"],
        train_image_dir=p["train_images"], val_image_dir=p["val_images"],
        test_image_dir=p["test_images"],
        train_transform=val_transform, val_transform=val_transform,
        image_size=config["data"]["image_size"],
        batch_size=1,  # batch size 1 for easy tracking
        num_workers=0,
        preprocess=config.get("preprocessing", {}).get("ben_graham", {}).get("enabled", True)
    )
    
    model = build_resnet50(
        num_classes=config["data"]["num_classes"],
        fc_hidden=config["model"]["fc_hidden"],
        dropout=config["model"]["dropout"],
        pretrained=config["model"]["pretrained"],
        unfreeze_blocks=config["model"]["unfreeze_blocks"],
    ).to(device)
    
    ckpt_path = Path(p["model_save_dir"]) / "best_model.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    else:
        print(f"  [ERROR] best_model.pth not found at {ckpt_path}.")
        return

    model.eval()
    
    failures = []
    
    print("  Evaluating test set for failures...")
    test_df = test_loader.dataset.df
    
    # ImageNet denormalisation constants
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits.float(), dim=1)
            pred   = logits.float().argmax(dim=1).item()
            label  = labels.item()
            prob   = probs[0, pred].item()
            
            if pred != label:
                failures.append({
                    "idx": i,
                    "id_code": test_df.iloc[i]["id_code"],
                    "true_label": label,
                    "pred_label": pred,
                    "prob": prob,
                    "tensor": images[0].cpu()
                })
                
    # Sort by highest confidence in the wrong prediction
    failures = sorted(failures, key=lambda x: x["prob"], reverse=True)[:10]
    
    if not failures:
        print("  No failures found!")
        return
        
    print(f"  Found {len(failures)} high-confidence failures. Generating Grad-CAMs...")
    
    # Generate Grad-CAM for these failures
    from src.explainability.gradcam import GradCAMVisualizer
    cam_viz = GradCAMVisualizer(model, device, method=config.get("explainability", {}).get("gradcam", {}).get("method", "gradcam"))
    
    out_dir = Path("presentation/failure_cases")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
    
    md_content = ["# Failure Case Report\n\nThis report highlights the top 10 most confident misclassifications by the model, using Grad-CAM to interpret why the model failed.\n\n"]
    
    for rank, fail in enumerate(failures):
        img_t = fail["tensor"]
        orig = img_t.permute(1, 2, 0).numpy()
        orig = (orig * std + mean).clip(0, 1).astype(np.float32)
        
        heatmap = cam_viz.explainer.generate_heatmap(img_t, target_class=fail["pred_label"])
        h, w = orig.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        overlay = cam_viz.explainer.overlay_heatmap(orig, heatmap_resized, alpha=0.5)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")
        
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title(f"Grad-CAM (for predicted: {class_names[fail['pred_label']]})", fontsize=12)
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=12)
        axes[2].axis("off")
        
        plt.tight_layout()
        img_filename = f"failure_{rank+1}_{fail['id_code']}.png"
        img_path = out_dir / img_filename
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        md_content.append(f"## {rank+1}. Patient `{fail['id_code']}`")
        md_content.append(f"- **True Grade:** {class_names[fail['true_label']]}")
        md_content.append(f"- **Predicted Grade:** {class_names[fail['pred_label']]} (Confidence: {fail['prob']:.2%})")
        md_content.append(f"\n![Failure {rank+1}](failure_cases/{img_filename})\n")
        md_content.append(f"**Analysis**: *[Add clinical interpretation of the Grad-CAM heatmap here. Does the model focus on artifacts?]*\n\n---\n")

    report_path = Path("presentation/Failure_Case_Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    print(f"  Report generated at {report_path}")

if __name__ == "__main__":
    main()
