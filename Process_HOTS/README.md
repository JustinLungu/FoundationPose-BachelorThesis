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








EXTRA INFO FOR DEVELOPER


Step 1: Understand the Goal
You want to run FoundationPose on the HOTS dataset instead of datasets like LineMOD or YCB. Here's a basic plan for this:

- Load the HOTS data (RGB images, instance masks).
- Modify FoundationPose's pipeline so it uses the RGB and instance masks for pose estimation.
- Test the new integration with HOTS

Step 2: What You Already Have

- FoundationPose code with existing scripts for LineMOD and YCB.
- HOTS dataset already downloaded and a script that loads it (hots.py).
- You’ve added a new run_hots.py script to integrate the HOTS data into FoundationPose.

Step 3: How to Connect HOTS to FoundationPose

- create run_hots.py
    - Load the HOTS dataset using the load_HOTS_scenes function from hots.py.
    - Pass the loaded images and masks to FoundationPose for pose estimation (using only the RGB and mask information).
- Modify FoundationPose for RGB-only Data
    - In FoundationPose's pose estimation (likely in estimater.py), the current method probably expects both RGB and depth data. However, HOTS only provides RGB and instance masks, so you need to modify the register() method in estimater.py to handle this:


Step 4: Keep It Simple for Now
- Ignore depth: HOTS doesn’t have depth, so your focus is on using the RGB and mask data for pose estimation.
- Modify only run_hots.py and register() in estimater.py to get started.
Once you can pass RGB and mask data from HOTS into the pose estimator and run it, you can improve the actual pose estimation method later.

Step 5: Next Steps
- Run the new script (run_hots.py) after these modifications.
- Test if the data is flowing correctly into the register() function from HOTS.
- Modify or improve the pose estimation logic for RGB-only data inside register().


https://github.com/gtziafas/HOTS
https://paperswithcode.com/dataset/linemod-1
https://www.ycbbenchmarks.com/#:~:text=YCB%20Object%20and%20Model%20Set,some%20widely%20used%20manipulation%20tests.
https://github.com/hz-ants/FFB6D?tab=readme-ov-file#datasets
https://github.com/ethnhe/PVN3D/tree/master





https://www.connectedpapers.com/main/dc4c9ae8c0cfc08ff6392aff69b0fd170da398a4/FoundationPose%3A-Unified-6D-Pose-Estimation-and-Tracking-of-Novel-Objects/graph
https://www.semanticscholar.org/paper/OnePose%3A-One-Shot-Object-Pose-Estimation-without-Sun-Wang/37f991349a7d389880d1ff0c62b248b64c296211
https://zju3dv.github.io/onepose_plus_plus/

