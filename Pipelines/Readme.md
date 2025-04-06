# 6D Object Pose & 3D Mesh Evaluation Pipelines

This repository contains four structured pipelines developed for evaluating and visualizing 6D pose estimation and 3D shape reconstruction. Each pipeline is modular, reproducible, and independently documented.

---

## Overview of Pipelines

### 1. **Process_HOTS**
- **Purpose:** Preprocesses the HOTS dataset into a structured format compatible with FoundationPose.
- **Capabilities:**
  - RGB-D image processing per object
  - Mesh loading and re-orientation
  - Scene and object-level YAML generation
  - Modular processors per modality (RGB, depth, mask)
- **Output:** Reformatted data folders with synthetic scenes and metadata

### 2. **Pose_Evaluation**
- **Purpose:** Evaluates 6D pose predictions against ground truth transforms.
- **Capabilities:**
  - Computes rotation, translation, pose, and ADD errors
  - Reformats LINEMOD-style ground truth into unified structure
  - Visualizes predictions (zoomed, annotated, orbit GIFs)
  - Outputs error plots and annotated overlays
- **Output:** Quantitative metrics and per-frame visual analysis

### 3. **Linemod_3D_noise**
- **Purpose:** Applies various geometric noise types to mesh models and compares them to the originals.
- **Capabilities:**
  - Adds Gaussian, Normal, Speckle, and Outlier noise
  - Compares noisy vs. clean meshes with per-vertex error stats
  - Exports CSV reports for each noise type
- **Output:** Corrupted meshes and error statistics for robustness testing

### 4. **3D_GenAI_Eval**
- **Purpose:** Evaluates AI-generated 3D models against ground truth meshes using geometric and perceptual metrics.
- **Capabilities:**
  - RANSAC + ICP mesh alignment
  - Metrics: Chamfer, Hausdorff, IoU, EMD, Curvature, Normal Consistency
  - Visual and JSON output for each model
- **Output:** Detailed results per model in `results/`, plus a global summary

---

## Common Features
- Modular, readable codebases with `main.py` entry points
- Centralized config/control for parameters
- Intermediate visual outputs (plots, overlays, 3D viewers)
- Designed for research use with extensibility in mind

---

## Directory Layout

```
Pipelines/
├── 3D_GenAI_Eval/           # Evaluates AI-generated meshes
├── Linemod_3D_noise/        # Corrupts meshes with geometric noise
├── Pose_Evaluation/         # Compares predicted 6D poses to GT
├── Process_HOTS/            # Converts HOTS dataset to FP format
└── Readme.md                # General overview (this file)
```

---

## Getting Started
Each folder contains its own README with:
- Dataset requirements
- Setup instructions
- How to run the pipeline
- Output description

Use `python main.py` in any pipeline folder to get started.

---

## Dependencies
Most pipelines rely on:
```bash
pip install open3d trimesh numpy matplotlib imageio PyYAML pillow scipy tqdm
```
See individual READMEs for additional tools per pipeline.

---

## Suggested Use Cases
- Training data preprocessing (e.g., HOTS → FoundationPose)
- Benchmarking 6D pose estimators
- Mesh degradation studies for robustness
- Quantitative evaluation of 3D generative models

---

## Notes
- All pipelines were tested with Python 3.9+ and Open3D 0.17+