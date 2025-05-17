#!/usr/bin/env python3
"""
Convert each GenAI OBJ under 'genAI_models' to PLY,
scaling them to match the original models via volume-based normalization.

Assumes:
  - genAI_models/<category>/model.obj
  - original_models/obj_XX.ply (ground-truth PLYs named by ID)

Outputs:
  - genAI_ply/obj_XX.ply
"""
import os
import sys
import shutil
import trimesh
import numpy as np

# Map folder names to original object IDs
CATEGORY_ID_MAP = {
    'camera':     4,
    'cat':        6,
    'drill':      8,
    'duck':       9,
    'egg_carton': 10,
    'gorilla':    1,
}

# Optional pre-scale rotations (radians)
ROT_X = 0.0
ROT_Y = 0.0
ROT_Z = 0.0


def safe_scale(mesh_gt, mesh_ai):
    """
    Compute uniform scale factor so that Mesh AI matches Mesh GT.
    1) If both are watertight volumes: scale by cube root of volume ratio.
    2) Else use convex hull volumes.
    3) Fallback to bounding-box diagonal ratio.
    """
    # attempt direct volumes
    if mesh_gt.is_volume and mesh_ai.is_volume and mesh_gt.volume > 0 and mesh_ai.volume > 0:
        return (mesh_gt.volume / mesh_ai.volume) ** (1/3)
    # fallback: convex-hull volumes
    vol_gt = mesh_gt.convex_hull.volume
    vol_ai = mesh_ai.convex_hull.volume
    if vol_gt > 0 and vol_ai > 0:
        return (vol_gt / vol_ai) ** (1/3)
    # last resort: bounding box diagonal ratio
    diag_gt = np.linalg.norm(mesh_gt.bounding_box.extents)
    diag_ai = np.linalg.norm(mesh_ai.bounding_box.extents)
    if diag_ai > 0:
        return diag_gt / diag_ai
    return 1.0


def process_models():
    base = os.path.dirname(os.path.realpath(__file__))
    src = os.path.join(base, 'genAI_models')
    dst = os.path.join(base, 'genAI_ply')
    orig_root = os.path.join(base, 'original_models')

    # Validate directories
    if not os.path.isdir(src):
        print(f"Error: genAI_models folder not found: {src}")
        sys.exit(1)
    if not os.path.isdir(orig_root):
        print(f"Error: original_models folder not found: {orig_root}")
        sys.exit(1)

    # Prepare output
    os.makedirs(dst, exist_ok=True)

    for category in sorted(os.listdir(src)):
        folder = os.path.join(src, category)
        obj_file = os.path.join(folder, 'model.obj')
        if not os.path.isfile(obj_file):
            print(f"Skipping '{category}': no model.obj")
            continue

        obj_id = CATEGORY_ID_MAP.get(category)
        if obj_id is None:
            print(f"Skipping '{category}': no ID mapping.")
            continue

        # ground-truth PLY path
        gt_ply = os.path.join(orig_root, f'obj_{obj_id:02d}.ply')
        if not os.path.isfile(gt_ply):
            print(f"Skipping ID {obj_id:02d}: ground-truth PLY not found.")
            continue

        print(f"Processing '{category}' → ID {obj_id:02d}...")
        # load meshes
        mesh_ai = trimesh.load(obj_file, force='mesh')
        mesh_gt = trimesh.load(gt_ply, force='mesh')
        if mesh_ai.is_empty or mesh_gt.is_empty:
            print("  Warning: empty mesh, skipping.")
            continue

        # center both
        mesh_ai.vertices -= mesh_ai.center_mass
        mesh_gt.vertices -= mesh_gt.center_mass
        # optional rotation
        if any((ROT_X, ROT_Y, ROT_Z)):
            from trimesh.transformations import euler_matrix
            R = euler_matrix(ROT_X, ROT_Y, ROT_Z)
            mesh_ai.apply_transform(R)
            mesh_gt.apply_transform(R)

        # compute and apply scale
        scale_factor = safe_scale(mesh_gt, mesh_ai)
        mesh_ai.apply_scale(scale_factor)
        print(f"  Applied scale ×{scale_factor:.4f}")

        # export
        out_name = f'obj_{obj_id:02d}.ply'
        out_path = os.path.join(dst, out_name)
        mesh_ai.export(out_path)
        print(f"  Exported to {out_path}\n")

if __name__ == '__main__':
    process_models()
