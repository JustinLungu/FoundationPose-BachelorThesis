#!/usr/bin/env python3
"""
Verify and visualize that the principal axes (x, y, z) of
original models match those of the GenAI-generated models, ensuring
consistent orientation for realistic comparison.

Now also prints the bounding-box diameters of each mesh for debugging scale.

Usage:
  python verify_alignment.py

Assumes:
  - 'original_models/obj_XX.ply'
  - 'genAI_ply/obj_XX.ply'

Outputs:
  - Printed diameters (original vs genAI) and axis deviations
  - Optional interactive overlay (red vs green)
"""
import os
import sys
import numpy as np

# Toggle visualization
VISUALIZE = True
# Threshold in degrees for misalignment warning
ANGLE_THRESHOLD = 5.0

# Check pillow for trimesh
try:
    from PIL import Image  # noqa: F401
except ModuleNotFoundError:
    print("Warning: 'pillow' not found. Visualization may be unavailable.")

try:
    import trimesh
except ModuleNotFoundError as e:
    print(f"Error importing trimesh: {e}. Install 'trimesh'.")
    sys.exit(1)


def compute_principal_axes(vertices):
    pts = vertices - vertices.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx]


def angle_between(v1, v2):
    cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosv = np.clip(abs(cosv), -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def verify_model(orig_path, gen_path):
    mesh_o = trimesh.load(orig_path, force='mesh')
    mesh_g = trimesh.load(gen_path, force='mesh')
    if mesh_o.is_empty or mesh_g.is_empty:
        print(f"Warning: empty mesh, skipping {os.path.basename(orig_path)}")
        return None
    axes_o = compute_principal_axes(mesh_o.vertices)
    axes_g = compute_principal_axes(mesh_g.vertices)
    angles = [angle_between(axes_o[:, i], axes_g[:, i]) for i in range(3)]
    return angles, mesh_o, mesh_g


def visualize_pair(mesh_o, mesh_g, title):
    mesh_o.visual.vertex_colors = [255, 0, 0, 100]
    mesh_g.visual.vertex_colors = [0, 255, 0, 100]
    scene = trimesh.Scene([mesh_o, mesh_g])
    try:
        scene.show(title=title)
    except Exception as e:
        print(f"Warning: Visualization skipped for {title}: {e}")


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    orig_dir = os.path.join(script_dir, 'original_models')
    gen_dir  = os.path.join(script_dir, 'genAI_ply')
    if not os.path.isdir(orig_dir) or not os.path.isdir(gen_dir):
        print("Error: 'original_models' and/or 'genAI_ply' directories not found.")
        sys.exit(1)

    print("Verifying model orientations and scales:\n")
    header = f"{'Model':<10}{'Orig_d':>8}{'Gen_d':>8}{'Axis1':>8}{'Axis2':>8}{'Axis3':>8}  Status"
    print(header)
    print('-' * len(header))

    for fname in sorted(os.listdir(orig_dir)):
        if not fname.lower().endswith('.ply'):
            continue
        orig_path = os.path.join(orig_dir, fname)
        gen_path  = os.path.join(gen_dir, fname)
        if not os.path.isfile(gen_path):
            print(f"{fname:<10} MISSING in genAI_ply")
            continue

        result = verify_model(orig_path, gen_path)
        if result is None:
            continue
        angles, mesh_o, mesh_g = result

        # compute diameters
        d_o = np.linalg.norm(mesh_o.bounding_box.extents)
        d_g = np.linalg.norm(mesh_g.bounding_box.extents)

        status = 'OK' if all(a <= ANGLE_THRESHOLD for a in angles) else 'MISALIGNED'
        print(f"{fname[:-4]:<10}{d_o:8.2f}{d_g:8.2f}{angles[0]:8.2f}{angles[1]:8.2f}{angles[2]:8.2f}  {status}")

        if VISUALIZE:
            print(f"  Visualizing {fname} (orig red, gen green)...")
            visualize_pair(mesh_o, mesh_g, title=f"{fname} alignment")

if __name__ == '__main__':
    main()
