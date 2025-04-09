# 3D GenAI Evaluation Pipeline

This pipeline evaluates AI-generated 3D meshes against ground-truth models using a comprehensive suite of geometric and perceptual metrics. It performs preprocessing, multi-step alignment, and quantitative evaluation, and outputs both visual and JSON-based reports.

---

## Overview

Given two folders containing:
- `.obj` files from a 3D generative model (`3d_data/ai_models/`)
- `.ply` ground truth meshes (`3d_data/gt_models/`)

the pipeline:
1. Loads and aligns each AI model with its corresponding ground truth
2. Preprocesses meshes for scale and centering
3. Applies RANSAC and multiscale ICP refinement
4. Computes evaluation metrics (IoU, Chamfer, Hausdorff, etc.)
5. Saves side-by-side visualizations and metric results per model

---

## Project Structure

```
3D_GenAI_Evaluation/
├── pipeline/
│   ├── constants.py             # Paths, thresholds, sampling configs
│   ├── loader.py                # Loads AI/GT meshes via Trimesh
│   ├── preprocessing.py         # Safe centering and scale normalization
│   ├── refiner.py               # Alignment via RANSAC + multiscale ICP
│   ├── visualizer.py            # 3D overlay visualization and saving
│   └── metrics/                 # Folder with all metric evaluators:
│       ├── base.py              # Abstract metric base class and classifier
│       ├── iou.py               # Boolean and voxel-based intersection-over-union
│       ├── chamfer.py           # Bi-directional Chamfer distance + visualization
│       ├── hausdorff.py         # Hausdorff max distance between surface samples
│       ├── normal_consistency.py# Cosine similarity between closest normals
│       ├── mean_curvature_error.py # Local curvature deviation at matched points
│       └── emd.py               # Earth Mover’s Distance (Hungarian algorithm)
│
├── main.py                     # Orchestrates the full evaluation pipeline
├── 3d_data/
│   ├── ai_models/               # Input .obj files (generated meshes)
│   └── gt_models/               # Ground truth .ply files
└── results/
    ├── <model_id>/             # One folder per evaluated model
    │   ├── before_scaling.png
    │   ├── after_scaling.png
    │   ├── after_icp.png
    │   ├── emd_vis.png
    │   ├── chamfer_error_vis.png
    │   ├── hausdorff_error_vis.png
    │   ├── normal_consistency_vis.png
    │   ├── mean_curvature_vis.png
    │   ├── mean_curvature_histogram.png
    │   ├── normal_angle_histogram.png
    │   └── metrics.json
    └── summary.json            # Summary across all models
```

---

## Evaluation Steps (Per Model)

1. **Load and Match:** Match `.obj` ↔ `.ply` by filename (order matters).
2. **Centering:** Translate both meshes to origin.
3. **Safe Scaling:** Match scale using volume or bounding box ratio.
4. **RANSAC + ICP:** Align with global (FPFH-based) and local (ICP) methods.
5. **Visualization:** Save visual overlays before/after scaling and after ICP.
6. **Metrics:** Run all metrics, assign class labels based on thresholds.
7. **Export:** Save metrics to JSON per model and one global summary.

---

## Metrics Overview

Each metric outputs a numeric **score** and a qualitative **class**:

### • IoU (Boolean & Voxel)
- **IoUBoolMetric**: Uses mesh volume intersection and union.
- **IoUVoxelMetric**: Voxels the shared space and compares occupancy.

### • Chamfer Distance
- Measures average squared distance from GT → AI and AI → GT.
- Visualized via colored scatter plots.

### • Hausdorff Distance
- Reports the maximum distance from any point in one mesh to the other.
- Sensitive to outliers.

### • Normal Consistency
- Computes angular similarity between nearest-point normals.
- Values closer to 1 mean better alignment of surface directions.

### • Mean Curvature Error
- Estimates curvature at each surface point, matches AI-GT points, and compares.
- Visualized with 3D color maps and histograms.

### • Earth Mover’s Distance (EMD)
- Uses optimal transport (Hungarian matching) between point sets.
- Visualizes matching lines and colored paths between GT-AI pairs.

Thresholds for “good”, “excellent”, or “warning” are defined in `constants.py`.

---

## How to Run

```bash
python main.py
```
Make sure your `.obj` files are in `3d_data/ai_models/` and `.ply` ground truths in `3d_data/gt_models/`.

---

## Dependencies

```bash
pip install trimesh open3d numpy matplotlib pillow scipy tqdm
```

---

## Notes
- Preprocessing uses robust volume checks and fallbacks to bounding box scaling
- All visual outputs use consistent red (GT) vs. green (AI) overlays
- Compatible with batched model evaluation
- `results/` folder is structured for easy per-model inspection