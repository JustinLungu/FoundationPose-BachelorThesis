# HOTS Pose Estimation Pipeline (FoundationPose Wrapper and Threestudio Auto-Generation Enabled)

This pipeline wraps the complete HOTS inference and evaluation flow using **FoundationPose**, adding support for **automatic 3D mesh generation** via **ThreeStudio** if a mesh is missing. It supports both **demo-style** and **LINEMOD-style** HOTS formats.

---

## Project Structure

```
project/
├── main.py
├── config.py
├── demo.py                  # For HOTS_Processed_demo/
├── linemod.py               # For HOTS_Processed_linemod/
├── generate_model.py        # Mesh generator (ThreeStudio wrapper)
├── object_mapping.py        # Maps LINEMOD ID ↔ name
├── results/                 # Output predictions and visualizations
├── data/
│   ├── HOTS_Processed_demo/
│   └── HOTS_Processed_linemod/
└── FoundationPose/          # External pose estimation framework
```

---

## Modes of Operation

Set mode in `config.py`:

```python
PIPELINE_MODE = "demo"  # or "linemod"
```

### Demo Mode

* One folder per object with RGB, depth, masks, and mesh
* If mesh is missing, `generate_model.py` triggers ThreeStudio mesh generation
* Supports:

  * Frame-wise registration
  * Pose tracking after first frame
  * Visualization overlays

### LINEMOD Mode

* Follows LINEMOD-style data layout
* Uses either GT or reconstructed meshes
* Supports different detection strategies:

  * Precise mask (`mask`)
  * Bounding box (`box`)
  * External detector (`detected`)
* Stores results per object and combined:

  * `linemod_res.yml`
  * `linemod_res_combined.yml`

---

## Mesh Generation (ThreeStudio Wrapper)

### `generate_model.py`

* When a mesh is missing in Demo mode, this module automatically:

  1. Runs `run_container.sh` with prompt = object name
  2. Locates `dreamfusion-sd/save/export*` folder
  3. Moves exported `.obj` mesh to `data/HOTS_Processed_demo/<object>/mesh/`

You can also run it directly:

```bash
python test.py  # Runs generator for "banana"
```

---

## Configuration

### DemoConfig (`config.py`)

```python
PROCESS_ALL_OBJECTS = False
CUSTOM_OBJECT_IDS = ["apple", "banana"]
USE_MASK_EVERY_FRAME = True  # full registration vs tracking
DEBUG_LEVEL = 2
SKIP_FRAMES_CONTAINING = ["kitchen"]
```

### LinemodConfig (`config.py`)

```python
PROCESS_ALL_OBJECTS = True
CUSTOM_OBJECT_IDS = [1, 20]
DETECT_TYPE = "mask"  # mask / box / detected
USE_RECONSTRUCTED_MESH = 1
```

---

## Outputs

### Demo Output (`results/demo_run/<object>/`):

* `ob_in_cam/<frame>.txt` → 4x4 pose matrix
* `track_vis/<frame>.png` → Visualization overlay

### LINEMOD Output (`results/linemod_run/`):

* `<object>/linemod_res.yml`
* `linemod_res_combined.yml`
* Optional visualizations if `DEBUG_LEVEL >= 3`

---

## How to Run

1. Place processed HOTS dataset under `data/`
2. Set mode in `config.py`
3. Run:

```bash
python main.py
```

> Meshes will be generated as needed (demo mode only).

---

## Dependencies

```
pip install numpy opencv-python imageio trimesh PyYAML
```

Also install FoundationPose and configure Docker + ThreeStudio for mesh generation.

---
