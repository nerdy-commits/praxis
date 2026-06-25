# =============================================================================
#  Praxis — Streamlit Dashboard
#  Explainable DR Grading + Patient Comorbidity Risk Network
# =============================================================================
"""
Run with:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
import os

# Ensure the root Praxis directory is in the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Change working directory to project root so all relative paths work in Docker
os.chdir(PROJECT_ROOT)

import cv2  # noqa: E402 — import after path setup
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Praxis — DR Risk Stratification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark background */
  [data-testid="stAppViewContainer"] { background: #0f0f1a; }
  [data-testid="stSidebar"]          { background: #16213e; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d5e;
    border-radius: 12px;
    padding: 1rem;
  }

  /* Headers */
  h1, h2, h3 { color: #e0e0ff; }
  p, li       { color: #b0b0d0; }

  /* Grade badges */
  .badge-0 { background:#1a4731; color:#2ecc71; padding:4px 12px; border-radius:20px; font-weight:600; }
  .badge-1 { background:#3d3500; color:#f1c40f; padding:4px 12px; border-radius:20px; font-weight:600; }
  .badge-2 { background:#4a2800; color:#e67e22; padding:4px 12px; border-radius:20px; font-weight:600; }
  .badge-3 { background:#4a0e0e; color:#e74c3c; padding:4px 12px; border-radius:20px; font-weight:600; }
  .badge-4 { background:#2d1040; color:#8e44ad; padding:4px 12px; border-radius:20px; font-weight:600; }

  /* Section divider */
  hr { border-color: #2d2d5e; }

  /* Sidebar radio */
  [data-testid="stSidebarNav"] a { color: #a0a0c0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GRADE_INFO = {
    0: ("No DR",           "🟢", "#2ecc71", "badge-0"),
    1: ("Mild DR",         "🟡", "#f1c40f", "badge-1"),
    2: ("Moderate DR",     "🟠", "#e67e22", "badge-2"),
    3: ("Severe DR",       "🔴", "#e74c3c", "badge-3"),
    4: ("Proliferative DR","🟣", "#8e44ad", "badge-4"),
}
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "clinical_metadata.csv"
METRICS_CSV  = PROJECT_ROOT / "outputs" / "results" / "test_metrics.csv"


# ── Cached model loader (prevents reloading on every page re-run) ─────────────
@st.cache_resource
def load_model():
    """Load the trained ResNet-50 once and cache it for the session."""
    checkpoint = OUTPUTS_DIR / "models" / "best_model.pth"
    if not checkpoint.exists():
        return None, None, None
    try:
        import torch
        from src.utils.config import load_config, get_device
        from src.models import build_resnet50
        cfg    = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
        device = get_device()
        model  = build_resnet50(
            num_classes=5,
            fc_hidden=cfg["model"]["fc_hidden"],
            dropout=cfg["model"]["dropout"],
            pretrained=False,
        ).to(device)
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, device, ckpt
    except Exception as e:
        return None, None, str(e)


# ── Cached data helpers ───────────────────────────────────────────────────────
@st.cache_data
def load_metadata():
    """Load clinical metadata CSV once and cache it."""
    if METADATA_CSV.exists():
        return pd.read_csv(METADATA_CSV)
    return None


@st.cache_data
def list_heatmap_files():
    """List Grad-CAM heatmap files once and cache the list."""
    gradcam_dir = OUTPUTS_DIR / "gradcam"
    if gradcam_dir.exists():
        return sorted([str(p) for p in gradcam_dir.glob("*.png")])
    return []


@st.cache_data(show_spinner=False)
def compute_graph_stats(gexf_path: str) -> dict:
    """
    Load patient graph and compute all network statistics.
    Cached so heavy computation only runs once per session.
    Uses approximate betweenness (k=50) to stay fast on dense graphs.
    """
    import networkx as nx
    import community as community_louvain

    G = nx.read_gexf(gexf_path)
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / max(len(degrees), 1)
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)

    # Louvain communities
    partition = community_louvain.best_partition(G, weight="weight")
    modularity = community_louvain.modularity(partition, G, weight="weight")
    n_clusters = len(set(partition.values()))

    # Centrality — approximate betweenness for speed (k=50 pivot nodes)
    bc = nx.betweenness_centrality(G, weight="weight", k=min(50, G.number_of_nodes()))
    dc = nx.degree_centrality(G)
    cc = nx.closeness_centrality(G)
    pr = nx.pagerank(G, weight="weight", alpha=0.85)

    return {
        "n_nodes":       G.number_of_nodes(),
        "n_edges":       G.number_of_edges(),
        "density":       density,
        "avg_degree":    avg_degree,
        "avg_clustering": avg_clustering,
        "modularity":    modularity,
        "n_clusters":    n_clusters,
        "partition":     partition,
        "degrees":       degrees,
        "bc":            bc,
        "dc":            dc,
        "cc":            cc,
        "pr":            pr,
        "node_attrs":    dict(G.nodes(data=True)),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Praxis")
    st.markdown("*Explainable DR Grading*  \n*+ Patient Risk Networks*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📊 Project Overview",
            "🖼️ DR Grading (CNN)",
            "🔍 Explainability (Grad-CAM)",
            "🌐 Patient Risk Network",
            "📈 Evaluation Metrics",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("IIIT Kottayam · Data Science Bootcamp 2026")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Project Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Project Overview":
    st.title("🔬 Praxis")
    st.subheader("Explainable Diabetic Retinopathy Grading with Patient Comorbidity Risk Networks")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🧠 CNN Classifier")
        st.markdown(
            "ResNet-50 fine-tuned on **APTOS 2019** for 5-class DR severity grading. "
            "Two-phase training: frozen backbone → selective fine-tuning of last 2 blocks."
        )
    with col2:
        st.markdown("### 🔍 XAI Layer")
        st.markdown(
            "**Grad-CAM** heatmaps highlight lesion regions driving each prediction — "
            "microaneurysms, hemorrhages, exudates — enabling clinical trust."
        )
    with col3:
        st.markdown("### 🕸️ Patient Network")
        st.markdown(
            "Cosine-similarity graph of patients built from clinical metadata. "
            "**Louvain clustering** surfaces high-risk comorbidity subgroups."
        )

    st.markdown("---")
    st.markdown("### System Architecture")
    st.code("""
    Retinal Fundus Image  +  Clinical Metadata
            │                       │
    Ben Graham Preprocessing    Normalization
            │                       │
    ResNet-50 (fine-tuned)     Feature Vectors
            │                       │
      5-class DR Grade ──────► Patient Similarity
            │                     Network (Cosine)
       Grad-CAM                       │
       Heatmaps              Louvain Clustering
                                       │
                             High-Risk Cluster IDs
    """, language="text")

    st.markdown("---")
    st.markdown("### Dataset")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**APTOS 2019 Blindness Detection**")
        class_data = {
            "Grade": ["0 — No DR", "1 — Mild", "2 — Moderate", "3 — Severe", "4 — Proliferative"],
            "Count": [1805, 370, 999, 193, 295],
            "Share (%)": [49.3, 10.1, 27.3, 5.3, 8.1],
        }
        st.dataframe(pd.DataFrame(class_data), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Tech Stack**")
        st.markdown("""
        | Layer | Library |
        |---|---|
        | Deep Learning | PyTorch, torchvision |
        | XAI | pytorch-grad-cam |
        | Network Science | NetworkX, python-louvain |
        | Visualization | matplotlib, plotly, pyvis |
        | Dashboard | Streamlit |
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DR Grading (CNN)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️ DR Grading (CNN)":
    st.title("🖼️ DR Grading — CNN Inference")
    st.markdown(
        "Upload a retinal fundus image to receive an automated DR severity grade "
        "with live **Grad-CAM** explainability."
    )
    st.markdown("---")

    # Pre-warm model cache BEFORE the file uploader renders.
    # This ensures the model is loaded once on page entry so that
    # uploading an image doesn't trigger a slow first-run + layout shift.
    _model_preload, _device_preload, _ckpt_preload = load_model()
    if _model_preload is not None:
        st.success("✅ Model loaded and ready for inference.")
    elif isinstance(_ckpt_preload, str):
        st.error(f"⚠️ Model load error: {_ckpt_preload}")
    else:
        st.info("ℹ️ No trained model checkpoint found. Running in demo mode.")

    # ── Controls row ─────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        uploaded_file = st.file_uploader(
            "Upload retinal fundus image (PNG / JPG)",
            type=["png", "jpg", "jpeg"],
            help="Images will be preprocessed with Ben Graham's method before inference.",
        )
    with ctrl_col2:
        cam_method = st.selectbox(
            "Grad-CAM Method",
            options=["gradcam", "gradcam++", "scorecam"],
            index=0,
            help="gradcam++ is sharper; scorecam is slowest but most accurate.",
        )
        cam_alpha = st.slider(
            "Heatmap opacity", min_value=0.2, max_value=0.8,
            value=0.5, step=0.05,
            help="Blend strength of the heatmap overlay.",
        )

    if uploaded_file is not None:
        col_img, col_pred = st.columns([1, 1])
        pred = None  # Initialize to avoid NameError if inference fails

        with col_img:
            st.subheader("Input Image")
            img_pil = Image.open(uploaded_file).convert("RGB")
            st.image(img_pil, use_container_width=True)

        with col_pred:
            st.subheader("Prediction")

            # Load model (cached — only runs once per session)
            model, device, ckpt = load_model()
            if model is not None:
                try:
                    import torch
                    from src.utils.config import load_config
                    from src.data import get_val_transforms

                    cfg = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))

                    # Preprocess with Ben Graham
                    from src.data.preprocessing import preprocess_image
                    img_np = np.array(img_pil)
                    if cfg.get("preprocessing", {}).get("ben_graham", {}).get("enabled", True):
                        sigma = cfg.get("preprocessing", {}).get("ben_graham", {}).get("sigma", 10)
                        img_np = preprocess_image(img_np, 224, sigma)
                    else:
                        img_np = cv2.resize(img_np, (224, 224))

                    transform = get_val_transforms(224)
                    augmented = transform(image=img_np)
                    tensor = augmented["image"].unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(tensor)
                        # Use .tolist() to avoid numpy binary compat issues
                        probs = torch.softmax(logits, dim=1).cpu().tolist()[0]
                        pred  = int(probs.index(max(probs)))

                    name, icon, color, badge = GRADE_INFO[pred]
                    st.markdown(f"<span class='{badge}'>{icon} {name}</span>",
                                unsafe_allow_html=True)
                    st.metric("Confidence", f"{probs[pred]*100:.1f}%")
                    st.metric("Checkpoint epoch", ckpt.get("epoch", "N/A"))

                    # Probability bars
                    st.markdown("**Class Probabilities**")
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(
                        x=probs,
                        y=[GRADE_INFO[i][0] for i in range(5)],
                        orientation="h",
                        marker_color=[GRADE_INFO[i][2] for i in range(5)],
                        text=[f"{p*100:.1f}%" for p in probs],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        xaxis=dict(range=[0, 1], showgrid=False),
                        yaxis=dict(autorange="reversed"),
                        height=250,
                        margin=dict(l=0, r=60, t=0, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e0e0ff"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Inference error: {e}")

            elif isinstance(ckpt, str):
                st.error(f"Model load error: {ckpt}")
            else:
                st.info("ℹ️ No trained model found. Run the pipeline first:\n\n```\npython main.py\n```")
                # Show demo placeholder
                import plotly.graph_objects as go
                demo_probs = [0.05, 0.08, 0.65, 0.15, 0.07]
                pred = 2
                name, icon, color, badge = GRADE_INFO[pred]
                st.markdown(f"**Demo prediction:** <span class='{badge}'>{icon} {name}</span>",
                            unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=demo_probs,
                    y=[GRADE_INFO[i][0] for i in range(5)],
                    orientation="h",
                    marker_color=[GRADE_INFO[i][2] for i in range(5)],
                    text=[f"{p*100:.1f}%" for p in demo_probs],
                    textposition="outside",
                ))
                fig.update_layout(
                    xaxis=dict(range=[0, 1], showgrid=False),
                    yaxis=dict(autorange="reversed"),
                    height=250, margin=dict(l=0, r=60, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0ff"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("*Demo mode — placeholder probabilities shown*")

        # ── Grad-CAM Heatmap (full-width below prediction columns) ───────────
        if model is not None and pred is not None:
            st.markdown("---")
            st.subheader("🔍 Grad-CAM Explanation")
            st.caption(
                f"Method: **{cam_method}** · Target class: **{GRADE_INFO[pred][0]}** (Grade {pred})"
            )

            with st.spinner("Generating Grad-CAM heatmap…"):
                try:
                    from src.explainability.gradcam import GradCAMExplainer

                    # Build explainer targeting the predicted class
                    target_layer = model.backbone.layer4[-1]
                    explainer = GradCAMExplainer(
                        model=model,
                        target_layer=target_layer,
                        device=device,
                        method=cam_method,
                    )

                    # Generate heatmap (the CAM library handles gradients internally)
                    heatmap = explainer.generate_heatmap(
                        tensor, target_class=pred
                    )  # shape (H, W), values [0,1]

                    # Prepare float RGB image [0,1] for overlay
                    orig_float = img_np.astype(np.float32)
                    if orig_float.max() > 1.0:
                        orig_float = orig_float / 255.0

                    h, w = orig_float.shape[:2]
                    heatmap_resized = cv2.resize(heatmap, (w, h))
                    overlay = explainer.overlay_heatmap(
                        orig_float, heatmap_resized, alpha=cam_alpha
                    )

                    # Apply JET colormap to raw heatmap for display
                    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
                    heatmap_jet   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    heatmap_jet_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

                    # Display 3 panels
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(
                            "<div style='text-align:center; color:#a0a0c0; "
                            "font-size:0.85rem; margin-bottom:4px'>📷 Preprocessed Input</div>",
                            unsafe_allow_html=True,
                        )
                        st.image(
                            (orig_float * 255).astype(np.uint8),
                            use_container_width=True,
                        )
                    with c2:
                        st.markdown(
                            "<div style='text-align:center; color:#a0a0c0; "
                            "font-size:0.85rem; margin-bottom:4px'>🌡️ Raw Heatmap (Jet)</div>",
                            unsafe_allow_html=True,
                        )
                        st.image(heatmap_jet_rgb, use_container_width=True)
                    with c3:
                        st.markdown(
                            "<div style='text-align:center; color:#a0a0c0; "
                            "font-size:0.85rem; margin-bottom:4px'>🔥 Overlay</div>",
                            unsafe_allow_html=True,
                        )
                        st.image(overlay, use_container_width=True)

                    # Legend
                    st.markdown(
                        "<div style='font-size:0.8rem; color:#808090; margin-top:6px'>"
                        "🔴 <b>Red/Yellow</b> = high activation (regions driving prediction) &nbsp;|&nbsp; "
                        "🔵 <b>Blue</b> = low activation"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                except Exception as cam_err:
                    st.warning(f"Grad-CAM could not be generated: {cam_err}")

    else:
        st.info("👆 Upload a retinal fundus image above to begin grading.")

        # Clinical grade reference
        st.markdown("---")
        st.markdown("### DR Grade Reference")
        cols = st.columns(5)
        for grade, (label, cols_i) in enumerate(zip(
            ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"], cols
        )):
            _, icon, color, _ = GRADE_INFO[grade]
            with cols_i:
                st.markdown(
                    f"<div style='text-align:center; padding:1rem; "
                    f"background:#1a1a2e; border-radius:10px; "
                    f"border:1px solid {color}33;'>"
                    f"<div style='font-size:2rem'>{icon}</div>"
                    f"<div style='color:{color}; font-weight:600'>Grade {grade}</div>"
                    f"<div style='color:#808090; font-size:0.8rem'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Explainability (Grad-CAM)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explainability (Grad-CAM)":
    st.title("🔍 Explainability — Grad-CAM Heatmaps")
    st.markdown(
        "Grad-CAM highlights the retinal regions that most influence each prediction. "
        "Clinically relevant structures (microaneurysms, hemorrhages, exudates) should "
        "appear as high-activation zones for severe grades."
    )
    st.markdown("---")

    gradcam_dir = OUTPUTS_DIR / "gradcam"
    heatmap_file_strs = list_heatmap_files()
    heatmap_files = [Path(p) for p in heatmap_file_strs]

    if heatmap_files:
        # Filter controls
        grade_filter = st.multiselect(
            "Filter by true DR grade",
            options=["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"],
            default=["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"],
        )
        correctness_filter = st.radio(
            "Show predictions",
            ["All", "Correct only", "Wrong only"],
            horizontal=True,
        )

        filtered = []
        for f in heatmap_files:
            name = f.stem
            grade_ok = any(g in name for g in grade_filter)
            if correctness_filter == "Correct only" and "correct" not in name:
                continue
            if correctness_filter == "Wrong only" and "wrong" not in name:
                continue
            if grade_ok:
                filtered.append(f)

        if filtered:
            n_cols = 3
            for i in range(0, len(filtered), n_cols):
                row_files = filtered[i:i + n_cols]
                cols = st.columns(n_cols)
                for col, fpath in zip(cols, row_files):
                    with col:
                        st.image(str(fpath), caption=fpath.stem, use_container_width=True)
        else:
            st.warning("No heatmaps match the selected filters.")

    else:
        st.info(
            "ℹ️ No Grad-CAM heatmaps found. Run the pipeline to generate them:\n\n"
            "```bash\npython main.py\n```\n\n"
            "Heatmaps will be saved to `outputs/gradcam/`."
        )
        st.markdown("---")
        st.markdown("### How Grad-CAM Works")
        st.markdown("""
        1. **Forward pass** through the trained ResNet-50
        2. **Gradients** of the target class score w.r.t. the last conv layer activations
        3. **Global average pooling** of gradients → importance weights per channel
        4. **Weighted sum** of activation maps → grayscale saliency map
        5. **Overlay** on original image using a JET colormap

        > 🔴 **Red/Yellow regions** = high importance for the prediction  
        > 🔵 **Blue regions** = low importance
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Patient Risk Network
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Patient Risk Network":
    st.title("🌐 Patient Comorbidity Risk Network")
    st.markdown(
        "Each node represents a patient. Edges connect patients with high **cosine similarity** "
        "across clinical features (HbA1c, BMI, BP, diabetes duration, comorbidities). "
        "**Louvain clustering** detects high-risk subgroups."
    )
    st.markdown("---")

    net_dir = OUTPUTS_DIR / "network"

    # ── Clinical cohort summary metrics ──────────────────────────────────────
    df = load_metadata()
    if df is not None:
        try:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Patients", len(df))
            if "dr_grade" in df.columns:
                severe = int((df["dr_grade"] >= 3).sum())
                c2.metric("Severe/Proliferative", severe,
                          delta=f"{severe/len(df)*100:.1f}% of cohort")
            if "hypertension" in df.columns:
                c3.metric("With Hypertension", int(df["hypertension"].sum()))
            if "hba1c" in df.columns:
                c4.metric("Mean HbA1c", f"{df['hba1c'].mean():.1f}%")
        except Exception:
            pass

    st.markdown("---")

    # ── Network Statistics Panel ──────────────────────────────────────────────
    gexf_path = net_dir / "patient_graph.gexf"
    _net_stats = {}  # shared across tabs

    if gexf_path.exists():
        st.subheader("\U0001f4ca Graph Statistics")
        try:
            with st.spinner("Computing graph statistics\u2026 (first visit only)"):
                _s = compute_graph_stats(str(gexf_path))

            _degrees     = _s["degrees"]
            _avg_degree  = _s["avg_degree"]
            _density     = _s["density"]
            _avg_clustering = _s["avg_clustering"]
            _partition   = _s["partition"]
            _modularity  = _s["modularity"]
            _n_clusters  = _s["n_clusters"]
            _bc = _s["bc"]
            _dc = _s["dc"]
            _cc = _s["cc"]
            _pr = _s["pr"]
            _node_attrs  = _s["node_attrs"]

            # ── Top-level graph metric cards ──────────────────────────────
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("\U0001f535 Nodes",        _s["n_nodes"])
            m2.metric("\U0001f517 Edges",        _s["n_edges"])
            m3.metric("📐 Density",      f"{_density:.4f}")
            m4.metric("🏘️ Clusters",     _n_clusters)
            m5.metric("⚡ Avg Degree",   f"{_avg_degree:.2f}")
            m6.metric("🔄 Avg Clustering", f"{_avg_clustering:.4f}")

            # Modularity badge
            mod_color = "#2ecc71" if _modularity > 0.3 else ("#f1c40f" if _modularity > 0.1 else "#e74c3c")
            st.markdown(
                f"<div style='background:#1a1a2e; border:1px solid {mod_color}; border-radius:8px;"
                f" padding:10px 16px; display:inline-block; margin-top:8px;'>"
                f"<span style='color:#a0a0c0; font-size:0.85rem;'>Modularity Q</span>"
                f"&nbsp;&nbsp;<span style='color:{mod_color}; font-size:1.4rem; font-weight:700;'>"
                f"{_modularity:.4f}</span>"
                f"&nbsp;&nbsp;<span style='color:#606080; font-size:0.8rem;'>"
                f"({'strong' if _modularity > 0.3 else 'moderate' if _modularity > 0.1 else 'weak'} community structure)"
                f"</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("")  # spacer

            # ── Centrality leaderboard table ──────────────────────────────
            with st.expander("🏆 Top-10 Centrality Leaderboard", expanded=True):
                top_n = 10
                top_nodes = sorted(_bc.items(), key=lambda x: x[1], reverse=True)[:top_n]
                cen_rows = []
                for rank, (node, bc_val) in enumerate(top_nodes, 1):
                    ndata = _node_attrs.get(node, {})
                    grade = int(ndata.get("dr_grade", -1)) if "dr_grade" in ndata else -1
                    grade_label = GRADE_INFO[grade][0] if 0 <= grade <= 4 else "\u2014"
                    cen_rows.append({
                        "Rank":               rank,
                        "Patient ID":         node,
                        "DR Grade":           grade_label,
                        "Betweenness ↑":      round(bc_val, 5),
                        "Degree Centrality ↑": round(_dc.get(node, 0), 5),
                        "Closeness ↑":        round(_cc.get(node, 0), 5),
                        "PageRank ↑":         round(_pr.get(node, 0), 5),
                    })
                st.dataframe(
                    pd.DataFrame(cen_rows).set_index("Rank"),
                    use_container_width=True,
                )
                st.caption(
                    "**Betweenness** — bridge patients connecting different risk subgroups.  "
                    "**Degree** — most connected (archetypal) patients.  "
                    "**Closeness** — patients central to the overall network.  "
                    "**PageRank** — patients most referenced by well-connected neighbours."
                )

            # ── Degree distribution mini-chart ────────────────────────────
            with st.expander("📈 Degree Distribution", expanded=False):
                import plotly.graph_objects as go
                deg_vals = sorted(_degrees.values(), reverse=True)
                fig_deg = go.Figure(go.Bar(
                    x=list(range(len(deg_vals))),
                    y=deg_vals,
                    marker_color="#6c5ce7",
                    marker_line_width=0,
                ))
                fig_deg.update_layout(
                    xaxis_title="Patient (sorted by degree)",
                    yaxis_title="Degree",
                    height=260,
                    margin=dict(l=0, r=0, t=10, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0ff"),
                )
                st.plotly_chart(fig_deg, use_container_width=True)

            # ── Cluster size distribution ─────────────────────────────────
            with st.expander("🏘️ Cluster Size Distribution", expanded=False):
                from collections import Counter
                cluster_sizes = Counter(_partition.values())
                sizes_sorted = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
                fig_cl = go.Figure(go.Bar(
                    x=[f"C{cid}" for cid, _ in sizes_sorted],
                    y=[sz for _, sz in sizes_sorted],
                    marker_color="#00cec9",
                    marker_line_width=0,
                ))
                fig_cl.update_layout(
                    xaxis_title="Community",
                    yaxis_title="Patients",
                    height=260,
                    margin=dict(l=0, r=0, t=10, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0ff"),
                )
                st.plotly_chart(fig_cl, use_container_width=True)

        except Exception as _gs_err:
            st.warning(f"Could not compute graph statistics: {_gs_err}")
    else:
        st.info("ℹ️ No patient graph found. Run the full pipeline (`python main.py`) to generate it.")

    st.markdown("---")

    # ── Visualization tabs ────────────────────────────────────────────────────
    tab_community, tab_grade, tab_interactive = st.tabs(
        ["🎨 Community View", "🩺 DR Grade View", "🔗 Interactive (PyVis)"]
    )

    with tab_community:
        img_path = net_dir / "patient_network_community.png"
        if img_path.exists():
            st.image(str(img_path), caption="Patient network coloured by Louvain community",
                     use_container_width=True)
        else:
            st.info("Run the pipeline to generate network visualisations.")

    with tab_grade:
        img_path = net_dir / "patient_network_drgrade.png"
        if img_path.exists():
            st.image(str(img_path), caption="Patient network coloured by predicted DR grade",
                     use_container_width=True)
            cols = st.columns(5)
            for grade, col in enumerate(cols):
                name, icon, color, _ = GRADE_INFO[grade]
                col.markdown(
                    f"<div style='text-align:center;'>"
                    f"<span style='color:{color}'>●</span> {name}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("Run the pipeline to generate network visualisations.")

    with tab_interactive:
        html_path = net_dir / "patient_network.html"
        if html_path.exists():
            st.components.v1.html(html_path.read_text(), height=700, scrolling=True)
        else:
            st.info(
                "Interactive HTML network not yet generated.\n\n"
                "You can generate it by running the network analysis step."
            )

    st.markdown("---")
    st.markdown("### Network Metrics Explained")
    st.markdown("""
    | Metric | Meaning |
    |---|---|
    | **Modularity (Q)** | Quality of community structure. Q > 0.3 indicates meaningful clustering. |
    | **DR Homophily** | Fraction of edges connecting patients with the same DR grade. High = natural grade-based clustering. |
    | **Degree Centrality** | Most connected "archetypal" patients — representative of their community. |
    | **Betweenness Centrality** | Bridge patients who connect different risk communities. |
    """)

    # Export button
    gexf_path = net_dir / "patient_graph.gexf"
    if gexf_path.exists():
        with open(gexf_path, "rb") as f:
            st.download_button(
                "⬇ Download GEXF (for Gephi)",
                f,
                file_name="patient_graph.gexf",
                mime="application/xml",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Evaluation Metrics":
    st.title("📈 Evaluation Metrics")
    st.markdown("---")

    # Load metrics if available
    if METRICS_CSV.exists():
        metrics_df = pd.read_csv(METRICS_CSV)
        row = metrics_df.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        qwk = row.get("quadratic_weighted_kappa", row.get("qwk", None))
        auc = row.get("auroc", row.get("auc_roc_macro", None))
        f1  = row.get("f1_weighted", row.get("f1_macro", None))
        acc = row.get("accuracy", None)

        if qwk is not None: c1.metric("QWK (primary)", f"{qwk:.4f}")
        if auc is not None: c2.metric("Macro AUC-ROC", f"{auc:.4f}")
        if f1  is not None: c3.metric("Weighted F1",   f"{f1:.4f}")
        if acc is not None: c4.metric("Accuracy",       f"{acc:.4f}")

    else:
        st.info("ℹ️ No results found. Train the model first: `python main.py`")

    # Baseline Model Comparison
    st.markdown("### 🏆 Baseline Model Comparison")
    baseline_csv = PROJECT_ROOT / "outputs" / "results" / "baseline_comparison.csv"
    baseline_img = PROJECT_ROOT / "outputs" / "figures" / "baseline_comparison.png"

    if baseline_csv.exists() and baseline_img.exists():
        col_table, col_chart = st.columns([1, 1.2])
        with col_table:
            st.markdown("Detailed evaluation scores compared against documented literature baselines:")
            baseline_df = pd.read_csv(baseline_csv, index_col=0)
            st.dataframe(baseline_df, use_container_width=True)
            st.caption("*Training from scratch suffers heavily on the relatively small ~3.6k image sample size, demonstrating the importance of ImageNet transfer learning.*")
        with col_chart:
            st.image(str(baseline_img), caption="Model Comparison — ResNet-50 vs Baselines", use_container_width=True)
    else:
        # Placeholder demo metrics
        st.markdown("### Expected Benchmark Metrics (APTOS 2019 literature)")
        bench_data = {
            "Model":       ["ResNet-50 (ours)",  "VGG-16 (baseline)", "From Scratch (baseline)"],
            "QWK ↑":       ["~0.88–0.92",         "~0.82–0.85",        "~0.65–0.75"],
            "AUC-ROC ↑":   ["~0.95–0.97",         "~0.92–0.94",        "~0.82–0.88"],
            "F1 Macro ↑":  ["~0.75–0.82",         "~0.68–0.74",        "~0.55–0.65"],
        }
        st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Confusion matrix
    cm_path = OUTPUTS_DIR / "figures" / "confusion_matrix.png"
    roc_path = OUTPUTS_DIR / "figures" / "roc_curves.png"
    curves_path = OUTPUTS_DIR / "figures" / "training_curves.png"

    if cm_path.exists() or roc_path.exists() or curves_path.exists():
        tabs = st.tabs(["Confusion Matrix", "ROC Curves", "Training Curves"])
        if cm_path.exists():
            with tabs[0]:
                st.image(str(cm_path), use_container_width=True)
        if roc_path.exists():
            with tabs[1]:
                st.image(str(roc_path), use_container_width=True)
        if curves_path.exists():
            with tabs[2]:
                st.image(str(curves_path), use_container_width=True)
    else:
        st.markdown("### Metric Definitions")
        st.markdown("""
        **Quadratic Weighted Kappa (QWK)**
        - Primary APTOS competition metric
        - Penalises predictions far from true grade more heavily than adjacent errors
        - Range: −1 to 1; > 0.8 = strong agreement

        **AUC-ROC (One-vs-Rest, macro)**
        - Measures discrimination ability across all 5 DR grades
        - Computed per class then macro-averaged

        **F1 Score (weighted)**
        - Harmonic mean of precision and recall
        - Weighted accounts for class imbalance

        **Confusion Matrix interpretation**
        - Diagonal = correct predictions
        - Off-diagonal = misclassifications
        - Adjacent-grade confusion (e.g., Mild ↔ Moderate) is clinically acceptable
        """)
