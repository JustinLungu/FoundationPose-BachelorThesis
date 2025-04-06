# Linemod_3D_Noise: Mesh Corruption and Comparison Pipeline

This pipeline applies multiple types of geometric noise to 3D mesh models (e.g., from LINEMOD), simulates real-world imperfections, and compares the noisy meshes against the originals using per-vertex distance statistics.

---

## Overview

The purpose of this pipeline is to:
- Simulate different types of realistic 3D noise (Gaussian, Normal, Speckle, Outlier)
- Create corrupted versions of each `.ply` or `.obj` mesh
- Compare each noisy mesh to its original and log statistical deviations (mean, std, min, max distance per vertex)

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

---

## Module Breakdown

### `main.py`
- Iterates over all noise types
- Calls `MeshProcessor` to apply noise and save results
- Calls `MeshComparer` to generate per-vertex distance statistics

### `mesh_processor.py`
Applies a noise strategy to all meshes in the input folder:
- Skips invalid meshes
- Saves results in the corresponding noise folder

### `comparer.py`
- Compares each noisy mesh to its clean version
- Computes:
  - Mean, Std, Min, Max of vertex displacements
- Outputs to a CSV file in each noise folder

---

## Noise Types

### `GaussianNoise`
Adds i.i.d. (independent and identically distributed) Gaussian noise to all vertices.
```python
vertices + N(mean, std_dev)
```

### `NormalNoise`
Adds noise along surface normals (directional displacement).

### `SpeckleNoise`
Applies multiplicative noise.
```python
vertices + (vertices * N(0, std_dev))
```

### `OutlierNoise`
Perturbs a small percentage of randomly chosen vertices.
- Controlled via `percentage` and `std_dev`

All noise classes implement the same `apply(mesh)` interface via `BaseNoise`.

---

## How to Run

1. Place original meshes into `models/` (as `.ply` or `.obj` files)
2. Run:
```bash
python main.py
```
3. Outputs will be saved in `models_noisy/<noise_type>/`
4. CSV reports will be named `comparison_<noise_type>.csv`

---

## Sample Output (Console)
```
--- Processing gaussian noise ---
Saved noisy mesh to: models_noisy/gaussian/obj_01.ply
Saved noisy mesh to: models_noisy/gaussian/obj_02.ply
...
Comparison results saved to: models_noisy/gaussian/comparison_gaussian.csv
...
```

---

## Dependencies
```bash
pip install open3d numpy
```

---

## Notes
- All `.ply` and `.obj` files are supported
- Compatible with downstream pipelines expecting clean + noisy mesh pairs
- Easily extendable by subclassing `BaseNoise` with new strategies
- Print statements provide clear progress tracking
