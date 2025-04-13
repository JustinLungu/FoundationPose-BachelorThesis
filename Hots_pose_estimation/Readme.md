# HOTS Pose Estimation Pipeline (FoundationPose Wrapper)

This pipeline evaluates 6D object pose estimation using the **FoundationPose** model on the **HOTS dataset**, supporting both **demo-style** and **LINEMOD-style** data layouts. It wraps the full workflow: reading RGB-D data, loading object meshes, predicting 6D poses, and saving both numerical results and visualization images.

---

## Project Structure

```
project/
├── main.py
├── config.py
├── demo.py                  # Inference on HOTS_Processed_demo
├── linemod.py               # Inference on HOTS_Processed_linemod
├── object_mapping.py        # ID ↔ name mapping for HOTS object categories
├── results/                 # Output predictions and visualizations
├── data/
│   ├── HOTS_Processed_demo/
│   └── HOTS_Processed_linemod/
└── FoundationPose/          # External pose estimation package
```

---

## Modes of Operation

Set the pipeline mode in `config.py`:
```
PIPELINE_MODE = "demo"  # or "linemod"
```

### Demo Mode
Processes HOTS objects in folder-per-object format (RGB, mask, depth, mesh).
- Supports per-frame registration or tracking
- Visualizes results with axis overlays and 3D bounding boxes

### LINEMOD Mode
Processes HOTS in LINEMOD-style format.
- Supports segmentation-based or bounding-box mask extraction
- Stores results as nested `linemod_res.yml` per object and `linemod_res_combined.yml`
- Ideal for benchmark-style evaluations

---

## Configuration Options (`config.py`)

### Demo Mode
```
USE_MASK_EVERY_FRAME = True        # full registration vs. tracking
ITERATION_REGISTER = 5
ITERATION_TRACK = 2
SKIP_FRAMES_CONTAINING = ["kitchen"]
CUSTOM_OBJECT_IDS = ["pringles_red", "pringles_hot"]
```

### LINEMOD Mode
```
PROCESS_ALL_OBJECTS = True
CUSTOM_OBJECT_IDS = [1, 20]
DETECT_TYPE = "mask"                # mask / box / detected
USE_RECONSTRUCTED_MESH = 0         # 0 = GT mesh, 1 = reconstructed
```

---

## Outputs

### Demo Mode Output (`results/demo_run/<object>/`):
- `track_vis/`: Frame-by-frame visualizations with predicted pose
- `ob_in_cam/`: 4x4 pose matrices per frame (`<frame_id>.txt`)

### LINEMOD Mode Output (`results/linemod_run/`):
- `<object>/linemod_res.yml`: Per-object pose predictions
- `linemod_res_combined.yml`: Global pose predictions across all objects
- Optional debug visualizations (`frame_<idx>_vis.png`) if `DEBUG_LEVEL >= 3`

---

## Core Components

### `main.py`
Selects between `DemoRunner` or `LinemodRunner` based on mode.

### `demo.py → DemoRunner`
- Loads HOTS demo-style RGB-D scenes and object meshes
- Performs frame-wise pose prediction (with tracking or mask-based registration)
- Visualizes and saves predictions

### `linemod.py → LinemodRunner`
- Processes objects by ID and directory layout
- Applies FoundationPose for registration
- Supports multiple detection types for masks
- Saves predictions as `linemod_res.yml` files

### `object_mapping.py`
Maps HOTS-style object IDs (integers) to descriptive names for logging and output clarity.

---

## How to Run

1. Set mode in `config.py`: `PIPELINE_MODE = "demo"` or `"linemod"`
2. Place HOTS dataset under `data/`
3. Run:
```
python main.py
```
4. Check `results/` for predictions and visualizations

---

## Notes
- Uses FoundationPose for both registration and optional refinement
- Requires `datareader`, `estimater`, and `dr.RasterizeCudaContext` from FoundationPose
- Skips frames with invalid depth/masks or excluded keywords
- Supports per-object or all-object inference via `CUSTOM_OBJECT_IDS`

---

## Dependencies

```
pip install numpy opencv-python imageio trimesh PyYAML
```
Also ensure FoundationPose and its dependencies are correctly installed.

---

© 2025 – HOTS Pose Estimation Inference Pipeline

