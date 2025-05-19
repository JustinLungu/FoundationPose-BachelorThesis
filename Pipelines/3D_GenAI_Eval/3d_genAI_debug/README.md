# 3D Model Alignment for GenAI Meshes (`3d_genAI`)

This directory contains tools to convert, align, and verify AI-generated 3D models against ground-truth Linemod models using scaling and geometric alignment (RANSAC + ICP). It supports integration with downstream pipelines like **FoundationPose**.

---

## Folder Structure

```
3d_genAI/
│
├── genAI_models/        # Input folders with OBJ+MTL+textures from GenAI (e.g., Magic123)
│   └── <category>/
│       ├── model.obj
│       ├── model.mtl
│       └── texture_kd.png
│
├── original_models/     # Ground-truth Linemod PLY models + models_info.yml
│   └── obj_XX.ply
│
├── genAI_ply/           # Output: Aligned GenAI models (PLY) + copied models_info.yml
│   └── obj_XX.ply
│
├── genAI_prep.py        # Main script: end-to-end conversion, scaling, alignment + visualization
├── obj_to_ply.py        # Quick script: converts & scales GenAI OBJs to PLYs (no alignment)
├── verify_alignment.py  # Diagnostic tool: verifies alignment + saves corrected models
```

---

## Setup

**Dependencies** (Python 3.8+ recommended):

* `numpy`
* `trimesh`
* `open3d`

Install with:

```
pip install trimesh open3d numpy
```

---

## Usage

### 1. End-to-End Conversion + Alignment + Visualization

```
python3 genAI_prep.py
```

This script:

* Converts each GenAI `.obj` to `.ply`
* Aligns to the corresponding Linemod model using RANSAC + ICP
* Prints bounding box diameter and axis deviations
* Saves aligned `.ply` into:

  * `genAI_ply/`
  * Also copied to: `../../FoundationPose/Linemod_preprocessed/genAI_ply/`
* Opens a visualization (red = GT, green = GenAI)

---

### 2. Quick OBJ → Scaled PLY (No Alignment)

```
python3 obj_to_ply.py
```

This is useful if you just want volume-matched `.ply` files from `.obj` (for quick inspection or intermediate processing).

---

### 3. Verify and Re-Align PLYs

```
python3 verify_alignment.py
```

* Compares `genAI_ply/obj_XX.ply` vs `original_models/obj_XX.ply`
* Aligns using RANSAC + ICP
* Reports bounding box diameter and angular deviation for 3 axes
* Saves aligned versions to `aligned_genAI_ply/`
* Visualizes results (optional)

---

## Notes

* Object categories are mapped using `CATEGORY_ID_MAP`:

  ```
  'gorilla': 1
  'camera': 4
  'cat': 6
  'drill': 8
  'duck': 9
  'egg_carton': 10
  ```
* All aligned models follow Linemod naming: `obj_XX.ply`.
* Angular deviation > 5° on any axis may suggest misalignment (reported in output).

---

## Example Output (`genAI_prep.py`)

```
Model      Orig_d   Gen_d   Axis1   Axis2   Axis3  Status
---------------------------------------------------------
cat          108.2   107.5     1.2     3.4     2.1  OK
duck          95.6    94.3     6.5     7.2     5.1  MISALIGNED
```

---

## Visual Output

When `VISUALIZE = True`, the scripts show:

* **Red**: Original (Linemod)
* **Green**: GenAI-aligned version

This helps diagnose orientation and scale issues interactively.
