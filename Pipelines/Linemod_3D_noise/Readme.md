# Linemod_3D_Noise: Mesh Corruption and Comparison Pipeline

This pipeline applies multiple types of geometric noise to 3D mesh models (e.g., from LINEMOD), simulates real-world imperfections, and compares the noisy meshes against the originals using per-vertex distance statistics.

---

## Overview

The purpose of this pipeline is to:
- Simulate different types of realistic 3D noise (Gaussian, Normal, Speckle, Outlier)
- Create corrupted versions of each `.ply` or `.obj` mesh
- Compare each noisy mesh to its original and log statistical deviations (mean, std, min, max distance per vertex)
- Each noise type is implemented as a separate class inheriting from `BaseNoise`, making it easy to add custom corruption types.

This is especially useful for robustness testing of 3D pipelines such as pose estimation or shape completion.

---

## Project Structure

```
Linemod_3D_Noise/
├── pipeline/
│   ├── noise/                  # Folder containing all noise strategies
│   │   ├── base_noise.py       # Abstract base for all noise classes
│   │   ├── gaussian.py         # Gaussian noise applied to all vertices
│   │   ├── normal.py           # Displacement along surface normals
│   │   ├── speckle.py          # Speckle noise: multiplicative perturbation
│   │   ├── outlier.py          # Outlier noise on a subset of vertices
│   ├── mesh_processor.py       # Applies noise strategies to input meshes
│   ├── comparer.py             # Compares original vs. noisy meshes
│
├── main.py                     # Pipeline entry point
├── models/                     # Folder with clean input meshes
└── models_noisy/               # Output: One subfolder per noise type
```

---

## Output Folder Structure

For each noise type, a subfolder is created under `models_noisy/`, e.g.:
```
models_noisy/
├── gaussian/
├── normal/
├── speckle/
└── outlier/
```
Each folder contains:
- Corrupted `.ply` files
- A CSV report (`comparison_<noise>.csv`) with:
  - filename
  - num_vertices
  - mean/std/min/max per-vertex distance from ground truth

> **Note**: The vertex distances in the CSV reports reflect the scale of your mesh. If using real-world scaled models (e.g., LINEMOD), distances are in **meters**.

- **Mesh Compatibility**: Both original and noisy meshes must have the **same number of vertices** and matching topology. Files with mismatched vertex counts are automatically skipped.
- **CSV filename**: `comparison_<noise>.csv` (e.g., `comparison_gaussian.csv`)
- **Output CSV includes per-vertex stats**: `mean_distance`, `std_distance`, `min_distance`, `max_distance`, and `num_vertices`

---

## Module Breakdown

### `main.py`

* Loops over all defined noise types in `NOISE_TYPES`
* For each noise type:

  * Applies the corresponding noise to all meshes using `MeshProcessor`
  * Saves noisy meshes under `models_noisy/<noise_name>/`
  * Compares noisy vs. clean meshes using `MeshComparer`
  * Saves per-vertex statistics as `comparison_<noise>.csv` in the same folder


### `mesh_processor.py`

* Applies a noise strategy to all `.ply` or `.obj` meshes in the `original_models/` folder
* Automatically creates a subfolder for the current noise type in `models_noisy/`
* Skips invalid or empty meshes
* Catches and logs processing errors per file
* `.obj` files are supported but treated as geometry-only — material/texture information is ignored

### `comparer.py`

* Compares each noisy mesh to its corresponding clean mesh based on per-vertex distances
* Only processes meshes with the same number of vertices and topology
* Computes per-vertex statistics:

  * Mean distance
  * Standard deviation
  * Minimum displacement
  * Maximum displacement
* Saves results to `models_noisy/<noise_type>/comparison_<noise>.csv`
* Logs warnings for missing or mismatched mesh pairs

### `base_noise.py`

* Defines an abstract base class `BaseNoise` that all noise strategies inherit from
* Each subclass must implement:

```python
class BaseNoise:
    def apply(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh
```

* Enables modular and extensible implementation of new noise strategies

---

## Noise Types

- All implementations assume the mesh is centered and scaled appropriately. Units follow Open3D’s convention — often meters for `.ply` models like LINEMOD.

Each noise type simulates different real-world corruption modes:

### `GaussianNoise`
Adds i.i.d. (independent and identically distributed) Gaussian noise to all vertices.
```
vertices + N(mean, std_dev)
```

### `NormalNoise`
Applies displacement in the direction of the surface normal:
```
vertices + (normal_direction * N(0, std_dev))
```

### `SpeckleNoise`
Applies multiplicative noise (values scaled by Gaussian factor):
```
vertices + (vertices * N(0, std_dev))
```

### `OutlierNoise`
Perturbs a small subset of randomly selected vertices:
```
selected_vertices += N(0, std_dev)  # applied to a percentage of vertices
```

---

## How to Run

The parameters (e.g., standard deviation, percentage of corrupted vertices) are hardcoded in `main.py` → `NOISE_TYPES` and can be easily tuned.

1. Place original meshes into `models/` (as `.ply` or `.obj` files)
2. Run the script:
```
python main.py
```
3. Outputs will be saved under `models_noisy/<noise_type>/`
4. Each folder will contain both noisy meshes and a `comparison_<noise>.csv` file

---

## Sample Output (Console)
```
--- Processing gaussian noise ---
Saved noisy mesh to: models_noisy/gaussian/obj_01.ply
Saved noisy mesh to: models_noisy/gaussian/obj_02.ply
...
Comparison results saved to: models_noisy/gaussian/comparison_gaussian.csv
```

---

## Dependencies
Install required packages:
```
pip install open3d numpy
```

---

## Notes
- Supports both `.ply` and `.obj` files
- CSV statistics help evaluate distortion effects
- Compatible with downstream pipelines expecting mesh pairs
- Easily extendable: create a new subclass of `BaseNoise`
- Clean console output for progress tracking and debugging

---

© 2025 – Linemod 3D Noise Simulation Pipeline

