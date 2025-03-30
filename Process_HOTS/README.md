# HOTS Dataset Processing Pipeline for FoundationPose

This repository provides a complete and modular pipeline to preprocess the HOTS dataset for use in 6D pose estimation frameworks such as **FoundationPose**.

## 🚀 Overview
The pipeline takes raw HOTS data (segmentation masks, RGB, depth images, and 3D object meshes) and transforms it into a structured format compatible with pose estimation training setups.

It includes:
- Folder structure creation
- RGB/mask/depth extraction and per-object organization
- Preprocessing and assignment of 3D meshes
- Final dataset summary for verification

---

## 📁 Directory Structure
```
HOTS_Processed/
├── apple/
│   ├── RGB/
│   ├── Depth/
│   ├── Mask/
│   ├── Mesh/
│   └── cam_K.txt
├── banana/
│   └── ...
...
```

---

## 🧱 Module Breakdown

### `main.py`
Entry point of the pipeline. It:
- Loops over all segmentation masks
- Matches masks with corresponding RGB and depth images
- Instantiates a scene processor
- At the end, triggers mesh normalization and dataset summary

### `manager.py`
Core orchestrator. The `HOTSProcessorManager` handles:
- Mapping mask labels to object names
- Creating object directories (if not already created)
- Saving:
  - Cropped RGB images
  - Binary masks
  - Scaled depth images
- Keeps a counter for each object (used in final summary)
- Triggers mesh processing at the end

### `structure.py`
Responsible for setting up the folder structure:
- Loads label mapping from `label_mapping.csv`
- Creates `RGB/`, `Depth/`, `Mask/`, and `Mesh/` subfolders
- Copies the `cam_K.txt` file into each object folder

### `processor_rgb.py`
Simple module to copy the RGB image to the object folder.

### `processor_mask.py`
Converts `.npy` semantic segmentation arrays into per-object binary masks.

### `processor_depth.py`
Handles depth:
- If a `.png` depth exists, it copies it
- If a `.npy` depth file exists, it converts it to 16-bit `.png`
- Normalizes float32 depth (if in meters) to millimeters

### `mesh_processor.py`
Processes and assigns 3D object models:
- Loads `.obj` files
- Rotates to match FoundationPose convention (Z+ forward)
- Scales mesh to a predefined max size
- Handles category sharing (e.g., all pringles variants get the same mesh)
- Copies associated `.mtl` and `.png` files (if found)
- Ignores missing files gracefully

### `base.py`
Defines a base class `ModalityProcessor` to standardize interface for RGB, depth, and mask processors.

---

## 🛠️ How to Run
1. Place HOTS raw dataset under `data/HOTS_v1/`
2. Make sure you have:
   - `label_mapping.csv`
   - `cam_K.txt`
   - Segmentation masks in `.npy` format
   - RGB images in `.png` format
   - Depth files (`.npy` or `.png`) in `data/depth/`
   - 3D models in `data/3D_models/<Object>/model.obj`
3. Run:
```bash
python main.py
```

---

## 📦 Output Summary
At the end of the run, a breakdown of how many images were processed per object is printed, e.g.:
```
Object processing summary:
 - apple: 24 image(s)
 - banana: 19 image(s)
 - book_blue: 25 image(s)
...
```

---

## ✅ Additional Features
- Mesh scaling is based on the longest bounding box edge (to preserve proportions)
- Avoids saving duplicated or unused mesh textures (like `model_0.png`)
- Gracefully skips missing RGB, depth, or mesh files with clear logging

---

## 📚 Dependencies
Make sure to install the following:
```bash
pip install numpy pandas opencv-python open3d imageio
```

---

## 📂 Notes
- The processed dataset is now ready to be plugged into a FoundationPose training loop
- You can tweak `target_dims` in `mesh_processor.py` to fit your object size preferences

---

## 🔧 Troubleshooting
- **Missing file warnings**: These don’t break the pipeline; they just mean a model/image was skipped.
- **Zero extent mesh**: Indicates the mesh is empty or corrupted — consider replacing it.
- **model_0.png showing up**: Now fixed by clearing internal Open3D textures.

---