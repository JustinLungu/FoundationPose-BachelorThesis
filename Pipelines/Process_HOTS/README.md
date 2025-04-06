# HOTS Dataset Processing Pipeline for FoundationPose

This repository provides a complete and modular pipeline to preprocess the HOTS dataset for use in 6D pose estimation frameworks such as **FoundationPose**.

## Overview
The pipeline takes raw HOTS data (segmentation masks, RGB, depth images, and 3D object meshes) and transforms it into a structured format compatible with pose estimation training setups.

It includes:
- Folder structure creation
- RGB/mask/depth extraction and per-object organization
- Preprocessing and assignment of 3D meshes
- Final dataset summary for verification

---

## Project Structure

This is how the repository is organized:

```
Process_HOTS/
├── hots_data/                    # Input data and assets
│   ├── 3D_models/                # Folder with category-wise mesh folders
│   ├── depth/                   # Optional .npy or .png depth images
│   ├── HOTS_v1/                 # Raw RGB and segmentation masks
│   └── cam_K.txt                # Shared intrinsics file
│
├── hots_pipeline/               # Main pipeline logic (modular)
│   ├── __init__.py
│   ├── base.py
│   ├── config.py
│   ├── manager.py
│   ├── mesh_processor.py
│   ├── processor_depth.py
│   ├── processor_mask.py
│   ├── processor_rgb.py
│   └── structure.py
│
├── main.py                      # Pipeline entry point
└── README.md
```

---

## Output Directory Structure

Depending on the setting in `config.py → FORMAT_TYPE`, the output structure will be:

### `"linemod"` format:
```
HOTS_Processed_linemod/
├── data/
│   ├── obj_01/
│   │   ├── rgb/
│   │   ├── depth/
│   │   ├── mask/
│   │   └── cam_K.txt
│   ├── obj_02/
│   │   └── ...
├── models/
│   ├── obj_01.ply
│   ├── obj_02.ply
│   └── ...
└── models.yml
```

### `"demo"` format:
```
HOTS_Processed_demo/
├── apple/
│   ├── RGB/
│   ├── Depth/
│   ├── Mask/
│   ├── Mesh/
│   └── cam_K.txt
├── banana/
│   └── ...
```

---

## Module Breakdown

### `main.py`
Entry point of the pipeline. It:
- Loops over all segmentation masks
- Matches masks with corresponding RGB and depth images
- Instantiates a scene processor
- At the end, triggers mesh normalization and dataset summary
- All paths and processing parameters are loaded from `hots_pipeline/config.py`

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

#### Mesh Alignment Logic

Each mesh is rotated by:

- **−90° about the X-axis**: Lays the object upright (if it was lying flat).
- **+180° about the Z-axis**: Ensures front-facing consistency across all models.

This transformation aligns the coordinate system so that:

- **+Z** → points forward
- **+Y** → points upward
- **+X** → points right (from the object’s perspective)

This rotation is applied using:
```
R_align = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, np.pi))
mesh.rotate(R_align, center=(0, 0, 0))
```
#### Mesh Scaling Logic
Each object mesh is uniformly scaled so its longest dimension matches a predefined target size (see `target_dims`). These values were chosen to reflect realistic object sizes in meters, ensuring dataset standardization and training consistency. See documentation for per-object rationales.

| Object         | Target Max Dimension (m) | Real-World Rationale                            |
|----------------|---------------------------|--------------------------------------------------|
| `apple`        | 0.08                      | Average apple diameter ~8 cm                     |
| `banana`       | 0.15                      | Typical banana length ~15 cm                     |
| `book`         | 0.22                      | Standard paperback/book width                    |
| `bowl`         | 0.19                      | Medium soup/cereal bowl diameter                 |
| `can`          | 0.12                      | 330ml soda can height                            |
| `cup`          | 0.11                      | Mug height ~11 cm                                |
| `fork`         | 0.19                      | Dinner fork length                               |
| `juice_box`    | 0.17                      | Small juice carton (200–250 ml)                  |
| `keyboard`     | 0.45                      | Full-sized keyboard width                        |
| `knife`        | 0.20                      | Medium kitchen knife length                      |
| `laptop`       | 0.33                      | 13–15 inch laptop diagonal                       |
| `lemon`        | 0.08                      | Slightly smaller than an apple                   |
| `marker`       | 0.15                      | Thick whiteboard marker                          |
| `milk`         | 0.24                      | 1L milk carton height                            |
| `monitor`      | 0.33                      | 24–27 inch monitor diagonal                      |
| `mouse`        | 0.11                      | Standard computer mouse                          |
| `orange`       | 0.08                      | Orange diameter ~8 cm                            |
| `peach`        | 0.08                      | Peach diameter similar to apple/orange           |
| `pear`         | 0.08                      | Normal size pear                                 |
| `pen`          | 0.15                      | Standard ballpoint pen length                    |
| `plate`        | 0.24                      | Dinner plate diameter                            |
| `pringles`     | 0.23                      | Pringles can height                              |
| `scissors`     | 0.17                      | Medium office scissors                           |
| `spoon`      ### `mesh_processor.py`
Processes and assigns 3D object models:
- Loads `.obj` files
- Rotates to match FoundationPose convention (Z+ forward)
- Scales mesh to a predefined max size (see `config.py → TARGET_DIMS`)
- Handles category sharing (e.g., all pringles variants get the same mesh)
- Copies associated `.mtl` and `.png` files (if found)
- Ignores missing files gracefully

#### Mesh Alignment Logic

Each mesh is rotated by:

- **−90° about the X-axis**: Lays the object upright (if it was lying flat).
- **+180° about the Z-axis**: Ensures front-facing consistency across all models.

This transformation aligns the coordinate system so that:

- **+Z** → points forward
- **+Y** → points upward
- **+X** → points right (from the object’s perspective)

This rotation is applied using:
```
R_align = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, np.pi))
mesh.rotate(R_align, center=(0, 0, 0))
```

#### Mesh Scaling Logic
Each object mesh is uniformly scaled so its longest dimension matches a predefined target size (see `config.py → TARGET_DIMS`). These values were chosen to reflect realistic object sizes in meters, ensuring dataset standardization and training consistency.

| Object         | Target Max Dimension (m) | Real-World Rationale                            |
|----------------|---------------------------|--------------------------------------------------|
| `apple`        | 0.08                      | Average apple diameter ~8 cm                     |
| `banana`       | 0.15                      | Typical banana length ~15 cm                     |
| `book`         | 0.22                      | Standard paperback/book width                    |
| `bowl`         | 0.19                      | Medium soup/cereal bowl diameter                 |
| `can`          | 0.12                      | 330ml soda can height                            |
| `cup`          | 0.11                      | Mug height ~11 cm                                |
| `fork`         | 0.19                      | Dinner fork length                               |
| `juice_box`    | 0.17                      | Small juice carton (200–250 ml)                  |
| `keyboard`     | 0.45                      | Full-sized keyboard width                        |
| `knife`        | 0.20                      | Medium kitchen knife length                      |
| `laptop`       | 0.33                      | 13–15 inch laptop diagonal                       |
| `lemon`        | 0.08                      | Slightly smaller than an apple                   |
| `marker`       | 0.15                      | Thick whiteboard marker                          |
| `milk`         | 0.24                      | 1L milk carton height                            |
| `monitor`      | 0.33                      | 24–27 inch monitor diagonal                      |
| `mouse`        | 0.11                      | Standard computer mouse                          |
| `orange`       | 0.08                      | Orange diameter ~8 cm                            |
| `peach`        | 0.08                      | Peach diameter similar to apple/orange           |
| `pear`         | 0.08                      | Normal size pear                                 |
| `pen`          | 0.15                      | Standard ballpoint pen length                    |
| `plate`        | 0.24                      | Dinner plate diameter                            |
| `pringles`     | 0.23                      | Pringles can height                              |
| `scissors`     | 0.17                      | Medium office scissors                           |
| `spoon`        | 0.19                      | Soup/dessert spoon                               |
| `stapler`      | 0.18                      | Typical desktop stapler length                   |



### `base.py`
Defines a base class `ModalityProcessor` to standardize interface for RGB, depth, and mask processors.

---

## Configuration

All paths and parameters (input folders, output format, mesh sizes, and rotation angles) are stored in:

```
hots_pipeline/config.py
```

Inside it, you can change:
- `BASE_DIR`, `DEPTH_DIR`, `MESH_DIR`: paths to input data
- `FORMAT_TYPE`: output format (demo or linemod)
- `TARGET_DIMS`: real-world scaling for each object
- `SHARED_CATEGORIES`: links object name prefixes to a shared mesh
- `ROTATION_X`, `ROTATION_Z`: rotation values to align meshes to FoundationPose

This makes the pipeline easy to adapt to other datasets or experiments.

---

## How to Run
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

## Output Summary
At the end of the run, a breakdown of how many images were processed per object is printed, e.g.:
```
Object processing summary:
 - apple: 24 image(s)
 - banana: 19 image(s)
 - book_blue: 25 image(s)
...
```

---

## Additional Features
- Mesh scaling is based on the longest bounding box edge (to preserve proportions)
- Avoids saving duplicated or unused mesh textures (like `model_0.png`)
- Skips missing RGB, depth, or mesh files with clear logging

---

## Dependencies
Make sure to install the following:
```bash
pip install numpy pandas opencv-python open3d imageio
```

---

## Notes
- The processed dataset is now ready to be plugged into a FoundationPose training loop
- You can customize all preprocessing logic in `config.py` to adapt for other datasets

---

## Troubleshooting
- **Missing file warnings**: These don’t break the pipeline; they just mean a model/image was skipped.
- **Zero extent mesh**: Indicates the mesh is empty or corrupted — consider replacing it.
- **model_0.png showing up**: Now fixed by clearing internal Open3D textures.

---








## EXTRA INFO FOR DEVELOPER


**Step 1: Understand the Goal**
You want to run FoundationPose on the HOTS dataset instead of datasets like LineMOD or YCB. Here's a basic plan for this:

- Load the HOTS data (RGB images, instance masks).
- Modify FoundationPose's pipeline so it uses the RGB and instance masks for pose estimation.
- Test the new integration with HOTS

**Step 2: What You Already Have**

- FoundationPose code with existing scripts for LineMOD and YCB.
- HOTS dataset already downloaded and a script that loads it (hots.py).
- You’ve added a new run_hots.py script to integrate the HOTS data into FoundationPose.

**Step 3: How to Connect HOTS to FoundationPose**

- create run_hots.py
    - Load the HOTS dataset using the load_HOTS_scenes function from hots.py.
    - Pass the loaded images and masks to FoundationPose for pose estimation (using only the RGB and mask information).
- Modify FoundationPose for RGB-only Data
    - In FoundationPose's pose estimation (likely in estimater.py), the current method probably expects both RGB and depth data. However, HOTS only provides RGB and instance masks, so you need to modify the register() method in estimater.py to handle this:


**Step 4: Keep It Simple for Now**
- Ignore depth: HOTS doesn’t have depth, so your focus is on using the RGB and mask data for pose estimation.
- Modify only run_hots.py and register() in estimater.py to get started.
Once you can pass RGB and mask data from HOTS into the pose estimator and run it, you can improve the actual pose estimation method later.

**Step 5: Next Steps**
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

