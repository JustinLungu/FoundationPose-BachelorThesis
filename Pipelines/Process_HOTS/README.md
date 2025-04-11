# HOTS Dataset Processing Pipeline for FoundationPose

This repository provides a complete, modular pipeline to preprocess the HOTS dataset for use in 6D pose estimation frameworks such as **FoundationPose**.

---

## Overview

The pipeline processes raw HOTS data—including segmentation masks, RGB images, depth maps, and 3D object meshes—and transforms them into a structured format compatible with pose estimation training.

### Main Features:
- Scene-level to object-level transformation
- RGB, depth, and mask extraction
- Per-object folder creation
- Mesh normalization and scaling
- Optional Linemod-style formatting
- Plug-and-play compatibility with FoundationPose

---

## Project Structure

```
Process_HOTS/
├── hots_data/
│   ├── 3D_models/                 # Input mesh directory
│   ├── depth/                    # Depth images (.npy or .png)
│   ├── HOTS_v1/                  # RGB + segmentation masks
│   │   └── scene/
│   │       ├── RGB/
│   │       └── SemanticSegmentation/SegmentationClass/
│   └── cam_K.txt                 # Shared camera intrinsics
│
├── hots_pipeline/               # Processing modules
│   ├── base.py
│   ├── config.py
│   ├── manager.py
│   ├── mesh_processor.py
│   ├── processor_depth.py
│   ├── processor_mask.py
│   ├── processor_rgb.py
│   └── structure.py
│
├── main.py                      # Entry point
└── README.md
```

---

## Output Directory Structure

### If `FORMAT_TYPE = "linemod"`
```
HOTS_Processed_linemod/
├── data/
│   ├── 01/
│   │   ├── rgb/
│   │   ├── depth/
│   │   ├── mask/
│   │   ├── info.yml
│   │   └── gt.yml
├── models/
│   ├── obj_01.ply
│   └── models_info.yml
```

### If `FORMAT_TYPE = "demo"`
```
HOTS_Processed_demo/
├── apple/
│   ├── rgb/
│   ├── depth/
│   ├── masks/
│   ├── mesh/
│   └── cam_K.txt
```

---

## Module Descriptions

### `main.py`
- Iterates through all segmentation mask files
- Loads corresponding RGB and depth images
- Calls `HOTSProcessorManager`
- Triggers mesh finalization and renaming (if Linemod)

### `manager.py`
- Maps object IDs to names via `label_mapping.csv`
- For each object in a scene:
  - Creates folders
  - Saves cropped RGB, binary mask, scaled depth
  - Updates `info.yml` and `gt.yml` (Linemod)
  - Tracks object counts
- Calls 3D mesh processor at the end

### `structure.py`
- Sets up output directories
- Copies `cam_K.txt` into each object folder
- Initializes YAML files (`info.yml`, `gt.yml`) if Linemod

### `processor_rgb.py`
- Copies RGB image

### `processor_mask.py`
- Extracts binary mask for given object label from `.npy`

### `processor_depth.py`
- Loads `.npy` or `.png` depth (depending on config)
- Converts to 16-bit PNG (millimeter scale)

### `mesh_processor.py`
- Loads `.obj` meshes
- Applies alignment:
```python
R_align = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, np.pi))
mesh.rotate(R_align, center=(0, 0, 0))
```
- Scales longest dimension to match target in `config.py → TARGET_DIMS`
- Shared mesh categories supported (e.g. all "pringles_*" share "Pringles/model.obj")
- Saves `model.obj` or `obj_xx.ply`
- Updates `models_info.yml` with inline bounding box metadata

### `base.py`
- Abstract base class for all modality processors (RGB, depth, mask)

---

## Configuration (`config.py`)
You can modify:
- `BASE_DIR`, `DEPTH_DIR`, `MESH_DIR`: Input paths
- `OUTPUT_DIR`: Target path
- `FORMAT_TYPE`: "linemod" or "demo"
- `TARGET_DIMS`: Scaling target sizes
- `SHARED_CATEGORIES`: Object prefix → Mesh folder
- `REQUIRE_DEPTH_NPY`: True to enforce `.npy` depth only
- `ROTATION_X`, `ROTATION_Y`, `ROTATION_Z`: Mesh alignment

---

## How to Run

1. Organize your files as shown above
2. Install dependencies:
```bash
pip install numpy pandas opencv-python open3d imageio trimesh pyyaml
```
3. Run the pipeline:
```bash
python main.py
```

---

## Output Summary Example
```
Object processing summary:
 - apple: 24 image(s)
 - banana: 19 image(s)
 - book_blue: 25 image(s)
```

---

## Developer Notes

### Integrating with FoundationPose
- Create `run_hots.py` to load HOTS data into FoundationPose
- Only use RGB + mask (ignore depth)
- Modify `register()` in FoundationPose to support RGB-only mode

### Tips:
- Test data flow first before optimizing pose logic
- All mesh transformations align to FoundationPose coordinate convention:
  - +Z → forward
  - +Y → upward
  - +X → right

---

## Troubleshooting
- Missing files → Gracefully skipped
- Zero-extent mesh → Warning; check source model
- `model_0.png` artifacts → Removed via Open3D cleanup

---

## Acknowledgements
- [HOTS Dataset](https://github.com/gtziafas/HOTS)
- [FoundationPose](https://github.com/ethnhe/FoundationPose)
- [Linemod Dataset](https://paperswithcode.com/dataset/linemod-1)
- [YCB Benchmark](https://www.ycbbenchmarks.com/)

---

© 2025




https://github.com/gtziafas/HOTS

https://paperswithcode.com/dataset/linemod-1

https://www.ycbbenchmarks.com/#:~:text=YCB%20Object%20and%20Model%20Set,some%20widely%20used%20manipulation%20tests.

https://github.com/hz-ants/FFB6D?tab=readme-ov-file#datasets

https://github.com/ethnhe/PVN3D/tree/master





https://www.connectedpapers.com/main/dc4c9ae8c0cfc08ff6392aff69b0fd170da398a4/FoundationPose%3A-Unified-6D-Pose-Estimation-and-Tracking-of-Novel-Objects/graph
https://www.semanticscholar.org/paper/OnePose%3A-One-Shot-Object-Pose-Estimation-without-Sun-Wang/37f991349a7d389880d1ff0c62b248b64c296211

https://zju3dv.github.io/onepose_plus_plus/

