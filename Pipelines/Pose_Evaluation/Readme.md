# Pose Evaluation and Visualization Pipeline for 6D Object Pose Estimation

This pipeline evaluates and visualizes pose estimation results (predicted vs. ground truth) using 4x4 transformation matrices, with outputs ranging from error metrics and plots to annotated images and orbiting GIFs.

---

## Overview

This modular pipeline takes YAML files containing predicted and ground truth transformations and:
- Computes quantitative errors (rotation, translation, pose, ADD)
- Visualizes per-frame alignment in 3D
- Generates interactive viewers, zoomed & full-frame images, annotated visuals, and orbiting GIFs
- Plots error trends, outliers, and distributions

---

## Project Structure

```
Pose_Eval/
├── pipeline/
│   ├── config.py               # All config variables: paths, zoom, thresholds, labels
│   ├── evaluation.py           # Core evaluator for pose errors and ADD
│   ├── formatter.py            # Reformatter for GT/pred YAML to unified structure
│   ├── visualizer.py           # Visual and statistical plot generation
│
├── main.py                     # Entry point for evaluation + visualization
└── plots/                      # Output plots and visualizations (auto-created)
```

---

## Input/Output Structure

### Input Files:
- `data/linemod_res.yml`: Raw predicted poses
- `data/gt.yml`: Raw ground truth poses
- `data/obj_01.ply`: Object mesh (used for ADD and visualization)

### Reformatted by:
- `formatter.reformat_predictions()`
- `formatter.reformat_ground_truth()`

### Output Files:
Created under the `plots/` directory:
```
plots/
├── frame_0_zoomed.png
├── frame_0_full.png
├── frame_0_annotated.png
├── orbit_animation.gif
├── error_outliers.png
├── error_trends.png
└── error_distributions.png
```

---

## Module Breakdown

### `main.py`
Main driver that:
- Reformats ground truth and prediction files
- Evaluates pose metrics (rotation, translation, Frobenius norm, ADD)
- Saves metric plots (outliers, trends, distributions)
- Visualizes one selected frame in multiple ways:
  - 3D overlay (interactive)
  - Static zoomed and full views
  - Annotated summary image
  - Orbiting camera GIF

### `evaluation.py`
Defines the `TransformationEvaluator` class:
- Loads 4x4 transformation matrices from YAML
- Loads object point cloud from `.ply`
- Computes:
  - **Rotation Error** (degrees)
  - **Translation Error** (meters)
  - **Pose Error** (Frobenius norm)
  - **ADD** (Average Distance of Model Points)

### `formatter.py`
Contains the `YAMLFormatter` class:
- **`reformat_predictions()`**: Renames frame keys and organizes predicted transforms
- **`reformat_ground_truth()`**: Converts LINEMOD-style `gt.yml` into a structured format compatible with evaluation

### `visualizer.py`
Includes two classes:
#### `TransformationVisualizer`
- Plots:
  - Error outliers
  - Trends over frames
  - Distributions (histograms)
- Uses `config.py → LABELS` and `TREND_THRESHOLDS` for styling

#### `AlignmentVisualizer`
- Shows side-by-side overlay of GT and predicted transformed point clouds
- Generates:
  - Static zoomed/full images
  - Annotated image with error metrics
  - Orbiting camera GIF
  - Interactive 3D view with optional azimuth/elevation config

---

## Configuration

Everything customizable is centralized in `config.py`:
- **Input paths**: GT/PRED YAMLs, object mesh
- **Zoom factors**: For zoomed/full/GIF views
- **Frame index**: For selecting a specific example to visualize
- **Rotation angles**: To align the base mesh if needed
- **Output paths**: Saved plots and images
- **Outlier/Trend thresholds**: For plots
- **Color/label map**: For consistent legends across plots

---

## How to Run

1. Place input files:
   - Raw prediction YAML at `data/linemod_res.yml`
   - Ground truth `gt.yml` at `data/gt.yml`
   - Mesh `.ply` at `data/obj_01.ply`

2. Run:
```bash
python main.py
```

3. Outputs will be saved under `plots/` as defined in `config.py`.

---

## Output Summary

Sample console output:
```
[✓] Reformatted prediction YAML saved as: reformatted/res_reformatted.yml
[✓] Reformatted GT YAML saved as: reformatted/gt_reformatted.yml
Rotation Error (deg): 3.2821
Translation Error (m): 0.0193
Pose Error (Frobenius norm): 0.4712
ADD (m): 0.0154
[✓] Saved image to plots/frame_0_zoomed.png
[✓] Saved image to plots/frame_0_full.png
[✓] Saved annotated image to plots/frame_0_annotated.png
[✓] Saved orbiting camera GIF to plots/orbit_animation.gif
[✓] Saved plot to plots/error_outliers.png
...
```

---

## Additional Features

- Supports arbitrary azimuth/elevation camera views (`show_interactive`)
- Annotated images include legend + error metrics
- All plots use consistent styling for reproducibility
- Color scheme:
  - **Red**: Ground truth
  - **Green**: Prediction
- Robust handling of YAML structure variations

---

## Dependencies

Install required libraries with:
```bash
pip install numpy open3d matplotlib imageio PyYAML pillow
```

---

## Troubleshooting

- **No transformations loaded**: Ensure YAML files contain 4x4 matrices under correct keys.
- **Empty `.ply` file or wrong format**: Make sure the object model has points.
- **Misaligned mesh view**: Adjust `ROTATION_ANGLES` in `config.py`.
- **Fonts not found for annotation**: The fallback font is used automatically.

