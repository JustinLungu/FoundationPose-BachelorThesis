# Pose Evaluation and Visualization Pipeline for 6D Object Pose Estimation

This pipeline evaluates and visualizes pose estimation results (predicted vs. ground truth) using 4x4 transformation matrices. It provides quantitative evaluation metrics, 3D alignment visualizations, and a variety of plots to support 6D pose estimation analysis.

---

## Overview

This modular pipeline:
- Reformats prediction and ground truth YAMLs into a unified structure
- Computes key pose errors (rotation, translation, Frobenius, ADD)
- Visualizes results for a selected frame in multiple ways:
  - Static overlay images (zoomed/full)
  - Annotated image with error metrics
  - Orbiting camera GIF
  - Interactive 3D view
- Plots error trends, outliers, and histograms for the entire dataset

---

## Project Structure

```
Pose_Eval/
├── pipeline/
│   ├── config.py               # Configuration for paths, zoom, thresholds, etc.
│   ├── evaluation.py           # Computes pose metrics (rotation, translation, pose, ADD)
│   ├── formatter.py            # Reformats YAMLs for unified evaluation
│   ├── visualizer.py           # Visual and statistical plot generation
│
├── main.py                     # Entry point
├── plots/                      # Auto-generated visualizations and plots
├── reformatted/                # Auto-generated formatted YAMLs
└── data/                       # Input GT/PRED YAMLs and .ply mesh file
```

---

## Input/Output Details

### Inputs
- `data/gt.yml`: Ground truth poses (LINEMOD format)
- `data/linemod_res.yml`: Predicted poses
- `data/obj_01.ply`: Object mesh for ADD and visualizations

### Outputs
- Reformatted YAMLs:
  - `reformatted/gt_reformatted.yml`
  - `reformatted/res_reformatted.yml`
- Visualizations (in `plots/`):
```
plots/
├── frame_0_zoomed.png
├── frame_0_full.png
├── frame_0_annotated.png
├── orbit_animation.gif
├── error_outliers.png
├── error_trends.png
├── error_distributions.png
```

---

## Module Breakdown

### `main.py`
- Orchestrates the full pipeline: formatting, evaluation, and visualization
- Uses `config.py` for paths and parameters

### `formatter.py`
- `reformat_predictions(input_file, output_file)`:
  - Renames frame keys to integers
- `reformat_ground_truth(input_file, output_file)`:
  - Converts LINEMOD-style GT format to nested YAMLs by object ID
  - **Assumption**: only one object per frame is used

> **Note**: This pipeline assumes **one object per frame** in the ground truth. If multiple objects exist per frame, `formatter.py` must be modified accordingly.

### `evaluation.py`
- Loads 4x4 matrices from formatted YAMLs
- Loads `.ply` object mesh and converts from **mm to meters**
- Computes:
  - **Rotation Error** (degrees)
  - **Translation Error** (meters)
  - **Pose Error** (Frobenius norm)
  - **ADD** (Average Distance of Model Points, in meters)

> **Unit Note**: ADD and Translation Error are calculated in **meters**.

### `visualizer.py`

#### `TransformationVisualizer`
- Plots errors across all frames:
  - Outliers
  - Trends
  - Distributions (histograms)
- Uses consistent colors and thresholds from `config.py → LABELS` and `TREND_THRESHOLDS`

#### `AlignmentVisualizer`
- 3D overlay of GT and predicted point clouds
- Generates:
  - Zoomed and full static images
  - Annotated image with per-frame errors
  - Orbiting GIF
  - Interactive viewer with optional azimuth/elevation angles

Example with azimuth and elevation:
```
visualizer_3d.show_interactive(frame_index=0, azimuth=45, elevation=30)
```

---

## Configuration (`config.py`)

- **Paths**: Input/output YAMLs, plots, mesh
- **Zoom levels**: Different factors for zoomed, full, GIF views
- **Frame index**: Which frame to visualize
- **Rotation angles**: To align the input mesh if needed
- **Thresholds**: For outlier classification and plotting
- **Label map**: Color/style info for plotting

Example snippet:
```
OUTLIER_THRESHOLDS = (10, 0.05, 0.1, 0.05)  # Rotation (deg), Translation (m), etc.
LABELS = [
    ("Rotation Error", "Degrees", "blue", "rotation"),
    ("Translation Error", "Meters", "orange", "translation"),
    ...
]
```

---

## How to Run

1. Place your input files in the `data/` folder:
   - `gt.yml`, `linemod_res.yml`, and `obj_01.ply`

2. Run the pipeline:
```
python main.py
```

3. All visual outputs will appear in `plots/`, and formatted YAMLs in `reformatted/`

---

## Sample Console Output
```
[✓] Reformatted prediction YAML saved as: reformatted/res_reformatted.yml
[✓] Reformatted GT YAML saved as: reformatted/gt_reformatted.yml
Rotation Error (deg): 3.2821
Translation Error (m): 0.0193
Pose Error (Frobenius norm): 0.4712
ADD (m): 0.0154
[✓] Saved image to plots/frame_0_zoomed.png
...
```

---

## Troubleshooting
- **No transformation matrices**: Ensure YAMLs use nested structure with valid 4x4 matrices
- **Misaligned visualizations**: Adjust `ROTATION_ANGLES` in `config.py`
- **Multiple objects per frame**: Requires changes to `formatter.py`
- **Empty mesh**: Check `.ply` file integrity

---

## Dependencies
Install all required libraries:
```
pip install numpy open3d matplotlib imageio PyYAML pillow
```

---

## Acknowledgements
- 6D pose metrics based on standard benchmarks like LINEMOD/YCB
- Mesh alignment and visualization via Open3D

---

© 2025 – Pose Evaluation Pipeline

