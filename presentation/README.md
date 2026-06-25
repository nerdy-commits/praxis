# Praxis Presentation & Deliverables

This directory contains the final portfolio and presentation assets for the Praxis project.

## Included Deliverables

- **demo.gif**: A compact visual walkthrough assembled from project outputs, highlighting model metrics, Grad-CAM examples, and patient network integration.
- **Ablation Studies**: Analysis (in notebooks or PDF format) showing performance differences with and without Ben Graham preprocessing and two-phase training.
- **Failure Case Report**: A structured report identifying misclassifications and analyzing their potential clinical impact using Grad-CAM explanations.
- **Deployed HTML Notebooks**: Static HTML exports of the six core project Jupyter Notebooks for easy reading without an active kernel.

## Generating HTML Notebooks

To generate static HTML versions of the notebooks, ensure `jupyter nbconvert` is installed, and run:

```bash
jupyter nbconvert --to html ../notebooks/*.ipynb --output-dir ./html_exports
```

## Creating the Demo GIF

1. Run the dashboard: `streamlit run app/streamlit_app.py`
2. Use a screen recording tool (e.g., LICEcap, ScreenToGif) to capture a 10-15 second flow: Upload image → View predictions → Examine Grad-CAM → Review patient risk clusters.
3. Save the output as `demo.gif` in this directory, replacing the compact generated GIF if a full dashboard recording is available.
