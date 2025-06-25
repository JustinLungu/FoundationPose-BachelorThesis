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
- Supports multi-method, multi-object evaluation via config-driven loops

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

* `linemod_results/ground_truth_pose/linemod_{obj_id:02d}.yml`: Ground truth poses (LINEMOD format) per object
* `linemod_results/pose_estimations/{method}/linemod_res_{obj_id}.yml`: Predicted poses for each method and object
* `linemod_results/original_models/obj_{obj_id:02d}.ply`: Object mesh used for ADD computation and visualization

### Outputs

* Reformatted YAMLs (per object and method):

  * `reformatted/gt_obj_{id}.yml`
  * `reformatted/{method}_obj_{id}_pred.yml`
* Visualizations (in `plots/{method}_obj_{id}/`):

```
plots/{method}_obj_{id}/
├── frame_0_zoomed.png
├── frame_0_full.png
├── frame_0_annotated.png
├── orbit_animation.gif
├── error_outliers.png
├── error_trends.png
├── error_distributions.png
```

* Summary:

  * `plots/results_summary.csv`: Consolidated mean metrics for all methods and objects


---

## Module Breakdown

### `main.py`
- Orchestrates the full pipeline: formatting, evaluation, and visualization
- Iterates over all configured objects and pose estimation methods
- Uses `config.py` for paths and parameters
- Saves visualizations per `{method}_obj_{id}` in subfolders under `plots/`

### `formatter.py`
- `reformat_predictions(input_file, output_file)`:
  - Renames frame keys to integers
- `reformat_ground_truth(input_file, output_file)`:
  - Converts LINEMOD-style GT format to a nested YAML format organized by object ID
  - Extracts 3×3 rotation and 3×1 translation vectors, converting translation from mm to meters
  - Constructs and stores 4×4 homogeneous transformation matrices

> **Note**: This pipeline assumes exactly one object per frame in the ground truth. The reformatted ground truth uses the object ID as the root key, with frame indices nested inside. If multiple objects exist per frame, `formatter.py` must be adapted accordingly.

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
- Interactive mode supports azimuth/elevation camera angles
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

* **Paths**:

  * `LINEMOD_ROOT`: Root folder containing GT, prediction results, and mesh data
  * Input/output YAMLs, plots, mesh files are all dynamically built from this base

* **Evaluation Targets**:

  * `POSE_METHODS`: List of pose estimation methods to evaluate (e.g., `original`, `3d_genAI`, etc.)
  * `OBJECT_IDS`: Object IDs to evaluate from the dataset

* **Zoom Levels**: Control camera distance for visualization

  * `ZOOMED_ZOOM_FACTOR`: Zoom for close-up image
  * `FULL_ZOOM_FACTOR`: Zoom for full-frame image
  * `GIF_ZOOM_FACTOR`: Zoom for orbiting GIF

* **Frame Index**:

  * `FRAME_IDX`: Index of the frame to visualize

* **Rotation Angles**:

  * `ROTATION_ANGLES`: Applied to input mesh (in degrees) to align before comparison

* **Thresholds**:

  * `OUTLIER_THRESHOLDS`: Thresholds for classifying outliers in plots
  * `TREND_THRESHOLDS`: Thresholds for trend/quality classification per metric

* **Label Map**:

  * `LABELS`: Defines each metric's name, unit, color, and internal key

Example snippet:

```python
OUTLIER_THRESHOLDS = (10, 0.05, 0.1, 0.05)  # Rotation (deg), Translation (m), etc.
LABELS = [
    ("Rotation Error", "Degrees", "blue", "rotation"),
    ("Translation Error", "Meters", "orange", "translation"),
    ("Pose Error", "Error", "green", "pose"),
    ("ADD Error", "Meters", "purple", "add"),
]
```

> **Tip**: You can control interactive rendering by toggling `SHOW_INTERACTIVE = True` in `config.py`.


---

## How to Run

To evaluate a single object/method pair, restrict `OBJECT_IDS` and `POSE_METHODS` in `config.py`

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
[✓] Reformatted prediction YAML saved as: reformatted/3d_genAI_obj_10_pred.yml
[✓] Reformatted GT YAML saved as: reformatted/gt_obj_10.yml

Results for 3d_genAI on object 10:
Rotation Error (deg): 4.1234
Translation Error (m): 0.0123
Pose Error (Frobenius norm): 0.4712
ADD (m): 0.0154

[✓] Saved image to plots/3d_genAI_obj_10/frame_0_zoomed.png
[✓] Saved image to plots/3d_genAI_obj_10/frame_0_full.png
[✓] Saved annotated image to plots/3d_genAI_obj_10/frame_0_annotated.png
[✓] Saved orbiting camera GIF to plots/3d_genAI_obj_10/orbit_animation.gif
[✓] Saved consolidated results to plots/results_summary.csv
```

> **Note**: The output will repeat for each object and method pair configured in `config.py`.


---

## Troubleshooting
- **No transformation matrices**: Ensure YAMLs use nested structure with valid 4x4 matrices
- **Misaligned visualizations**: Adjust `ROTATION_ANGLES` in `config.py`
- **Multiple objects per frame**: Requires changes to `formatter.py`
- **Empty mesh**: Check `.ply` file integrity
- **Missing predicted YAMLs**: The pipeline skips them gracefully and prints a warning
- **Empty GT or Prediction Matrices**: Raises ValueError during visualization

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

