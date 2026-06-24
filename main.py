# =============================================================================
#  Main Entry Point — Full Pipeline
# =============================================================================
"""
Run the complete Praxis pipeline end-to-end:
    1. Load data and create DataLoaders
    2. Train CNN classifier (ResNet-50)
    3. Evaluate on held-out test set
    4. Generate Grad-CAM explanations
    5. Build & analyse patient similarity network

Usage:
    python main.py
    python main.py --config configs/default.yaml
    python main.py --skip-train   # skip training, load saved checkpoint
    python main.py --skip-network # skip network analysis step
"""

import argparse
import sys
import os
from pathlib import Path

# Force UTF-8 on Windows console to avoid cp1252 UnicodeEncodeError
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config, get_device
from src.data import get_train_transforms, get_val_transforms
from src.data.dataset import create_data_loaders
from src.models import build_resnet50, Trainer
from src.evaluation import compute_all_metrics, plot_confusion_matrix, plot_roc_curves
from src.utils.visualization import plot_training_curves, plot_class_distribution
from torch.amp import autocast


def parse_args():
    parser = argparse.ArgumentParser(description="Praxis -- DR Grading Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; load best_model.pth from model_save_dir"
    )
    parser.add_argument(
        "--skip-network", action="store_true",
        help="Skip patient similarity network step"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Train VGG-16 and from-scratch baselines and save comparison table"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -- Load Config ----------------------------------------------------------
    config = load_config(args.config)
    device = get_device()
    paths  = config["paths"]
    data_cfg  = config["data"]
    model_cfg = config["model"]
    xai_cfg   = config["explainability"]
    net_cfg   = config["network"]

    print("=" * 60)
    print("  Praxis -- Explainable DR Grading Pipeline")
    print(f"  Device: {device}")
    print("=" * 60)

    # -- Step 1: Data ---------------------------------------------------------
    print("\n>> Step 1: Loading data...")
    train_transform = get_train_transforms(data_cfg["image_size"], config.get("augmentation", {}))
    val_transform   = get_val_transforms(data_cfg["image_size"])

    train_loader, val_loader, test_loader = create_data_loaders(
        train_csv=paths["train_csv"],
        val_csv=paths["val_csv"],
        test_csv=paths["test_csv"],
        train_image_dir=paths["train_images"],
        val_image_dir=paths["val_images"],
        test_image_dir=paths["test_images"],
        train_transform=train_transform,
        val_transform=val_transform,
        image_size=data_cfg["image_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 4),
        preprocess=config.get("preprocessing", {}).get("ben_graham", {}).get("enabled", True),
    )
    print(f"  Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # Class distribution plot (from loaded dataset)
    plot_class_distribution(
        train_loader.dataset.get_class_distribution(),
        save_path=f"{paths['figures_dir']}/class_distribution.png"
    )

    # -- Step 2: Train Model --------------------------------------------------
    model = build_resnet50(
        num_classes=data_cfg["num_classes"],
        fc_hidden=model_cfg["fc_hidden"],
        dropout=model_cfg["dropout"],
        pretrained=model_cfg["pretrained"],
        unfreeze_blocks=0
        if model_cfg.get("freeze_strategy") in {"full", "partial"}
        else model_cfg["unfreeze_blocks"],
    ).to(device)

    checkpoint_path = Path(paths["model_save_dir"]) / "best_model.pth"

    if args.skip_train and checkpoint_path.exists():
        print(f"\n>> Step 2: Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded epoch {ckpt['epoch']} | Val QWK: {ckpt['val_qwk']:.4f}")
        history = None
    else:
        print("\n>> Step 2: Training ResNet-50...")
        # Compute inverse-frequency class weights from training set
        class_weights = train_loader.dataset.get_class_weights()
        print(f"  Class weights: {class_weights.numpy().round(3)}")
        trainer = Trainer(model, device, config, save_dir=paths["model_save_dir"])
        # Resume from best checkpoint if it exists (avoids restarting from scratch)
        resume_ckpt = checkpoint_path if checkpoint_path.exists() else None
        if resume_ckpt:
            print(f"  [RESUME] Found checkpoint -- continuing from saved state")
        history = trainer.train(
            train_loader, val_loader,
            class_weights=class_weights,
            resume_from=str(resume_ckpt) if resume_ckpt else None,
        )
        plot_training_curves(history, f"{paths['figures_dir']}/training_curves.png")

        # Reload best checkpoint for evaluation
        if checkpoint_path.exists():
            print(f"  [INFO] Reloading best checkpoint from {checkpoint_path} for evaluation...")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

    # -- Step 3: Evaluate -----------------------------------------------------
    print("\n>> Step 3: Evaluating on test set...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
            probs  = torch.softmax(logits.float(), dim=1)
            preds  = logits.float().argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    print("\n  Test Metrics:")
    for name, value in metrics.items():
        if value is not None:
            print(f"    {name}: {value:.4f}")

    plot_confusion_matrix(all_labels, all_preds, f"{paths['figures_dir']}/confusion_matrix.png")
    plot_roc_curves(all_labels, all_probs, f"{paths['figures_dir']}/roc_curves.png")

    # Save metrics to CSV
    metrics_path = Path("outputs/results")
    metrics_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_path / "test_metrics.csv", index=False)
    print(f"  Metrics saved -> outputs/results/test_metrics.csv")

    # -- Step 4: Grad-CAM -----------------------------------------------------
    print(f"\n>> Step 4: Generating Grad-CAM heatmaps ({xai_cfg['num_samples']} samples)...")
    try:
        from src.explainability.gradcam import GradCAMVisualizer
        # Clear VRAM used by the evaluation pass before allocating for Grad-CAM
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cam_viz = GradCAMVisualizer(
            model, 
            device,
            method=xai_cfg.get("method", "gradcam")
        )
        cam_viz.batch_generate(
            test_loader,
            n_samples=xai_cfg["num_samples"],
            output_dir=paths["gradcam_dir"],
        )
        print(f"  Grad-CAM images saved -> {paths['gradcam_dir']}")
    except Exception as e_gpu:
        print(f"  [INFO] GPU Grad-CAM failed ({e_gpu}), retrying on CPU...")
        try:
            from src.explainability.gradcam import GradCAMVisualizer
            model_cpu = model.cpu()
            cam_viz   = GradCAMVisualizer(model_cpu, torch.device("cpu"), method=xai_cfg.get("method", "gradcam"))
            cam_viz.batch_generate(
                test_loader,
                n_samples=xai_cfg["num_samples"],
                output_dir=paths["gradcam_dir"],
            )
            model.to(device)   # restore to GPU for any further steps
            print(f"  Grad-CAM images saved (CPU) -> {paths['gradcam_dir']}")
        except Exception as e_cpu:
            print(f"  [WARN] Grad-CAM skipped entirely: {e_cpu}")

    # -- Step 5: Patient Similarity Network -----------------------------------
    if not args.skip_network:
        metadata_path = Path(paths["metadata_csv"])
        if not metadata_path.exists():
            print(f"\n>> Step 5: Metadata not found at {metadata_path}.")
            print("  Run: python scripts/download_data.py  to generate metadata.")
        else:
            print("\n>> Step 5: Building patient similarity network...")
            try:
                from src.network.analysis import NetworkAnalyzer

                clinical_df  = pd.read_csv(metadata_path)
                feature_cols = net_cfg["clinical_features"]

                id_col = "patient_id" if "patient_id" in clinical_df.columns else clinical_df.columns[0]
                
                # CNN -> network integration: replace synthetic grades with test set predictions
                print("  Updating patient metadata with CNN predictions...")
                test_df = test_loader.dataset.df
                test_preds = dict(zip(test_df["id_code"], all_preds))
                if "dr_grade" in clinical_df.columns:
                    clinical_df["dr_grade"] = clinical_df[id_col].map(test_preds).fillna(clinical_df["dr_grade"]).astype(int)
                
                dr_grades = clinical_df["dr_grade"].values if "dr_grade" in clinical_df.columns else None

                analyzer = NetworkAnalyzer(
                    clinical_df=clinical_df,
                    feature_columns=[c for c in feature_cols if c in clinical_df.columns],
                    threshold=net_cfg["edge_threshold"],
                    dr_grades=dr_grades,
                )

                summary = analyzer.summary()
                print(f"\n  Network Summary:")
                for k, v in summary.items():
                    if v is not None:
                        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

                # Save visualisations
                net_dir = paths["network_dir"]
                viz_cfg  = net_cfg["visualization"]
                analyzer.visualize_network(
                    save_path=f"{net_dir}/patient_network_community.png",
                    color_by="community", size_by="degree",
                    layout=viz_cfg["layout"],
                )
                if dr_grades is not None:
                    analyzer.visualize_network(
                        save_path=f"{net_dir}/patient_network_drgrade.png",
                        color_by="dr_grade", size_by="betweenness",
                        layout=viz_cfg["layout"],
                    )
                if viz_cfg.get("export_gexf", True):
                    analyzer.export_gexf(f"{net_dir}/patient_graph.gexf")

                # Interactive PyVis HTML
                try:
                    analyzer.export_pyvis(f"{net_dir}/patient_network.html")
                except Exception as _e:
                    print(f"  [WARN] PyVis skipped: {_e}")

                # High-risk clusters
                high_risk = analyzer.get_high_risk_clusters()
                print(f"\n  High-risk communities (sorted by mean DR grade):")
                for comm_id, stats in list(high_risk.items())[:5]:
                    print(f"    Community {comm_id}: size={stats['size']}  "
                          f"mean_grade={stats['mean_dr_grade']:.2f}  "
                          f"severe_frac={stats['severe_fraction']:.2%}")

                print(f"\n  Network outputs saved -> {net_dir}")

            except Exception as e:
                print(f"  [WARN] Network analysis skipped: {e}")
    else:
        print("\n>> Step 5: Network analysis skipped (--skip-network flag).")

    # -- Step 6 (optional): Baseline comparison --------------------------------
    if args.baseline:
        _run_baselines(config, device, test_loader, paths)

    # -- Done -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline complete! Check outputs/ for all results.")
    print("  Launch dashboard:  streamlit run app/streamlit_app.py")
    print("=" * 60)


def _run_baselines(config, device, test_loader, paths):
    """
    Train VGG-16 (pretrained) and a ResNet-50 from scratch, evaluate both,
    then produce a 3-model comparison table alongside the main ResNet-50.
    """
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.data import get_train_transforms, get_val_transforms
    from src.data.dataset import create_data_loaders
    from src.models import build_resnet50, Trainer
    from src.models.resnet_classifier import build_vgg16
    from src.evaluation import compute_all_metrics

    print("\n" + "=" * 60)
    print("  BASELINE COMPARISON: VGG-16 + ResNet-50 from scratch")
    print("=" * 60)

    data_cfg  = config["data"]
    model_cfg = config["model"]
    p         = paths

    # Shared loaders
    train_transform = get_train_transforms(data_cfg["image_size"])
    val_transform   = get_val_transforms(data_cfg["image_size"])
    train_loader, val_loader, _ = create_data_loaders(
        train_csv=p["train_csv"], val_csv=p["val_csv"], test_csv=p["test_csv"],
        train_image_dir=p["train_images"], val_image_dir=p["val_images"],
        test_image_dir=p["test_images"],
        train_transform=train_transform, val_transform=val_transform,
        image_size=data_cfg["image_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 2),
    )
    class_weights = train_loader.dataset.get_class_weights()

    baseline_configs = [
        ("VGG-16",        build_vgg16(num_classes=5, pretrained=True),  "vgg16_model.pth"),
        ("ResNet50-Scratch", build_resnet50(num_classes=5, pretrained=False), "scratch_model.pth"),
    ]

    all_results = {}

    # Load existing ResNet-50 result
    metrics_csv = Path("outputs/results/test_metrics.csv")
    if metrics_csv.exists():
        row = pd.read_csv(metrics_csv).iloc[0]
        all_results["ResNet-50"] = {
            "QWK":      round(float(row.get("quadratic_weighted_kappa", 0)), 4),
            "AUC-ROC":  round(float(row.get("auroc_weighted", 0)), 4),
            "F1-Macro": round(float(row.get("f1_macro", 0)), 4),
            "Accuracy": round(float(row.get("accuracy", 0)), 4),
        }
        print(f"  ResNet-50 (main): loaded from {metrics_csv}")

    for name, model, ckpt_name in baseline_configs:
        print(f"\n  Training {name}...")
        model = model.to(device)
        trainer = Trainer(model, device, config,
                          save_dir=str(Path(p["model_save_dir"]) / "baselines"))
        trainer.train(train_loader, val_loader, class_weights=class_weights)

        # Evaluate on test set
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
        all_results[name] = {
            "QWK":      round(metrics.get("quadratic_weighted_kappa", 0), 4),
            "AUC-ROC":  round(metrics.get("auroc_weighted", 0), 4),
            "F1-Macro": round(metrics.get("f1_macro", 0), 4),
            "Accuracy": round(metrics.get("accuracy", 0), 4),
        }
        print(f"  {name} -> QWK={all_results[name]['QWK']:.4f}  "
              f"AUC={all_results[name]['AUC-ROC']:.4f}")

        # Save model
        torch.save(model.state_dict(),
                   Path(p["model_save_dir"]) / "baselines" / ckpt_name)

    # Save comparison CSV
    comp_df = pd.DataFrame(all_results).T
    comp_csv = Path("outputs/results/baseline_comparison.csv")
    comp_csv.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(comp_csv)
    print(f"\n  Comparison table:\n{comp_df.to_string()}")
    print(f"  Saved -> {comp_csv}")

    # Bar chart
    fig_path = Path(p["figures_dir"]) / "baseline_comparison.png"
    ax = comp_df.plot(kind="bar", figsize=(10, 5), colormap="viridis", edgecolor="white")
    ax.set_title("Model Comparison — Evaluation Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Chart saved -> {fig_path}")


if __name__ == "__main__":
    main()
