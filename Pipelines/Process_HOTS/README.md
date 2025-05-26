# HOTS Dataset Processing Pipeline for FoundationPose

This repository offers a modular and robust pipeline to convert the **HOTS dataset** into formats compatible with 6D pose estimation frameworks like **FoundationPose**. It handles everything from raw input segmentation masks to final structured datasets with processed RGB, depth, and mesh data.

---

## Purpose

Pose estimation models like FoundationPose often assume structured datasets (e.g., Linemod). HOTS, by contrast, provides RGB and semantic segmentation data at a scene level. This pipeline bridges that gap:

- Extracts object-level data from scene-level segmentation
- Aligns and scales 3D meshes
- Structures data into Linemod-style or Demo-style outputs
- Enables rapid training and benchmarking

---

## Key Features

- Format-agnostic: supports `demo` or `linemod` output
- Modular architecture: clean separation of logic per modality (RGB, depth, mask, mesh)
- Realistic scaling: meshes scaled to real-world dimensions
- Semantic-aware: label mapping from segmentation to class name
- Compatible structure: ready-to-use for FoundationPose

---

## Project Structure

```
Process_HOTS/
├── hots_data/
│   ├── 3D_models/                      # Meshes per category (e.g., Book/model.obj)
│   ├── depth/                         # Depth data (.npy/.png)
│   ├── HOTS_v1/
│   │   └── scene/
│   │       ├── RGB/                  # RGB images
│   │       └── SemanticSegmentation/SegmentationClass/  # Segmentation masks
│   └── cam_K.txt                     # Shared intrinsics
│
├── hots_pipeline/
│   ├── base.py                       # Abstract interface for modality processors
│   ├── config.py                     # Configurable parameters
│   ├── manager.py                    # Processing coordinator
│   ├── mesh_processor.py            # Mesh loading, rotation, scaling
│   ├── processor_depth.py           # Depth conversion
│   ├── processor_mask.py            # Mask extraction per label
│   ├── processor_rgb.py             # RGB copying logic
│   └── structure.py                 # Directory setup utilities
│
├── main.py                          # Entry point for processing
└── README.md
```

---

## Output Formats

Set `FORMAT_TYPE` in `config.py` to choose between:

### `linemod` Format
```
HOTS_Processed_linemod/
├── data/
│   ├── 01/
│   │   ├── rgb/       # Cropped RGB
│   │   ├── depth/     # Depth in PNG (uint16)
│   │   ├── mask/      # Binary mask
│   │   ├── info.yml   # Intrinsics
│   │   └── gt.yml     # Ground truth poses
├── models/
│   ├── obj_01.ply     # Processed mesh
│   └── models_info.yml
```

### `demo` Format
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
- Reads `.npy` segmentation masks
- Finds matching RGB/depth files
- Calls `HOTSProcessorManager` per scene
- Calls mesh finalization and renaming (if `linemod`)

### `manager.py`
- Loads label mapping
- For each object in a mask:
  - Saves RGB, mask, and depth
  - Adds entry in `info.yml` and `gt.yml` (for `linemod`)
  - Tracks image counts per object

### `structure.py`
- Creates the output directory layout depending on format
- Copies `cam_K.txt`
- Initializes empty YAML files for `linemod`

### `processor_rgb.py`
- Simple `shutil.copy()` of the RGB input image

### `processor_mask.py`
- Extracts binary mask from `.npy` file
- Outputs 255-valued PNG mask per object

### `processor_depth.py`
- Accepts `.npy` or `.png` depth
- Normalizes float32 (meters) to uint16 PNG (millimeters)
- Supports config flag to require `.npy`

### `mesh_processor.py`
- Loads `.obj` mesh and centers it
- Applies alignment:
```python
R_align = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, np.pi))
mesh.rotate(R_align, center=(0, 0, 0))
```
- Uniformly scales mesh so that its largest dimension matches `TARGET_DIMS[obj]`
- Handles category merging (e.g., all "pringles_*" → Pringles folder)
- Writes `.obj` (for demo) or `.ply` (for linemod)
- Updates `models_info.yml` with diameter and bounding box stats

### `base.py`
- Defines abstract class `ModalityProcessor`

---

## Configuration Options (`config.py`)

```python
FORMAT_TYPE = "demo"  # or "linemod"
REQUIRE_DEPTH_NPY = True
SKIP_IMAGES_CONTAINING = []  # Keywords to exclude from filenames

TARGET_DIMS = {
  "apple": 0.08,
  "banana": 0.15,
  ... # real-world scale for each object
}
SHARED_CATEGORIES = {
  "pringles": "Pringles",
  "pen": "Pen",
  ...
}
ROTATION_X = -3.14159 / 2  # Lay upright
ROTATION_Z = 3.14159       # Face forward
```

---

## Running the Pipeline

1. Prepare files:
   - Place masks in `.npy` under `hots_data/HOTS_v1/scene/SemanticSegmentation/SegmentationClass/`
   - Place RGB in `.png` under `RGB/`
   - Place depth in `.npy` or `.png` under `depth/`
   - Ensure mesh files are at `3D_models/<Object>/model.obj`
   - Include `label_mapping.csv` and `cam_K.txt`

2. Install requirements:
```bash
pip install numpy pandas opencv-python open3d imageio trimesh pyyaml
```

3. Run:
```bash
python main.py
```

---

## Example Output Summary
```
=== Processing Summary ===
 - apple: 24 image(s)
 - banana: 19 image(s)
 - keyboard: 15 image(s)
```

---

## Developer Notes: Integrating HOTS with FoundationPose

### Step 1: Load the HOTS dataset
- Create `run_hots.py`
- Use your existing loader (`hots.py`) to load RGB and masks

### Step 2: Modify FoundationPose
- Edit `estimater.py → register()` to accept only RGB + masks
- Skip any depth-dependent logic initially

### Step 3: Run and iterate
- Ensure data is passed from HOTS into the pipeline correctly
- Once validated, optimize the pose logic and add depth back later if needed

---

## Troubleshooting
- Missing files: Logged with warnings but skipped safely
- Zero-extent mesh: May need replacement or fixing
- Open3D texture warnings: Cleaned by clearing mesh textures

---

## References
- [HOTS GitHub](https://github.com/gtziafas/HOTS)
- [FoundationPose Paper](https://www.connectedpapers.com/main/dc4c9ae8c0cfc08ff6392aff69b0fd170da398a4/FoundationPose)
- [YCB Dataset](https://www.ycbbenchmarks.com)
- [Linemod Benchmark](https://paperswithcode.com/dataset/linemod-1)

---

© 2025 – HOTS Processing Framework



https://github.com/gtziafas/HOTS

https://paperswithcode.com/dataset/linemod-1

https://www.ycbbenchmarks.com/#:~:text=YCB%20Object%20and%20Model%20Set,some%20widely%20used%20manipulation%20tests.

https://github.com/hz-ants/FFB6D?tab=readme-ov-file#datasets

https://github.com/ethnhe/PVN3D/tree/master





https://www.connectedpapers.com/main/dc4c9ae8c0cfc08ff6392aff69b0fd170da398a4/FoundationPose%3A-Unified-6D-Pose-Estimation-and-Tracking-of-Novel-Objects/graph
https://www.semanticscholar.org/paper/OnePose%3A-One-Shot-Object-Pose-Estimation-without-Sun-Wang/37f991349a7d389880d1ff0c62b248b64c296211

https://zju3dv.github.io/onepose_plus_plus/

