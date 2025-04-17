#!/usr/bin/env python3
"""
Convert each generated OBJ model in the '3d_genAI' folder to PLY,
scale them based on the `diameter` entries in 'genAI_models/models_info.yml',
and save the resulting PLY files into the 'genAI_models' folder.
"""
import os
import sys
import yaml
import trimesh
import numpy as np
import shutil

# --- Configuration ---
# Map from folder name to object ID (matches original_models entries)
CATEGORY_ID_MAP = {
    'camera': 4,
    'cat': 6,
    'drill': 8,
    'duck': 9,
    'egg_carton': 10,
    'gorilla': 1,
}
# Optional alignment rotations (in radians)
ROT_X = 0.0
ROT_Y = 0.0
ROT_Z = 0.0


def process_models():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    src_root = os.path.join(script_dir, 'genAI_models')
    ply_root = os.path.join(script_dir, 'genAI_ply')
    orig_info = os.path.join(script_dir, 'original_models', 'models_info.yml')

    # Validate directories
    if not os.path.isdir(src_root):
        print(f"Error: Source folder not found: {src_root}")
        sys.exit(1)
    if not os.path.isfile(orig_info):
        print(f"Error: models_info.yml not found in original_models.")
        sys.exit(1)

    # Load size info
    with open(orig_info, 'r') as f:
        models_info = yaml.safe_load(f) or {}

    # Prepare output
    os.makedirs(ply_root, exist_ok=True)
    shutil.copy2(orig_info, os.path.join(ply_root, 'models_info.yml'))
    print(f"Copied models_info.yml to {ply_root}")

    # Process each category folder
    for name in sorted(os.listdir(src_root)):
        folder = os.path.join(src_root, name)
        if not os.path.isdir(folder):
            continue
        obj_file = os.path.join(folder, 'model.obj')
        if not os.path.isfile(obj_file):
            print(f"Skipping '{name}': no model.obj found.")
            continue

        print(f"Processing '{name}'...")
        mesh = trimesh.load(obj_file, force='mesh')
        if mesh.is_empty:
            print(f"  Warning: '{name}' mesh is empty, skipping.")
            continue

        # Center and rotate
        mesh.vertices -= mesh.center_mass
        if any((ROT_X, ROT_Y, ROT_Z)):
            transform = trimesh.transformations.euler_matrix(ROT_X, ROT_Y, ROT_Z)
            mesh.apply_transform(transform)

        # Compute current diameter
        bounds = mesh.bounding_box.bounds
        size = bounds[1] - bounds[0]
        current_d = np.linalg.norm(size)

        # Scale to target diameter
        obj_id = CATEGORY_ID_MAP.get(name)
        target_d = None
        if obj_id and obj_id in models_info and 'diameter' in models_info[obj_id]:
            target_d = float(models_info[obj_id]['diameter'])
        if target_d:
            scale_factor = target_d / current_d
            mesh.apply_scale(scale_factor)
            print(f"  Scaled: {current_d:.2f} → {target_d:.2f} (×{scale_factor:.4f})")
        else:
            print(f"  No target diameter for '{name}', skipping scale.")

        # Export as obj_ID.ply with zero-padded two-digit ID
        if obj_id is None:
            print(f"  Error: No ID mapping for '{name}', skipping export.")
            continue
        filename = f"obj_{obj_id:02d}.ply"
        out_file = os.path.join(ply_root, filename)
        mesh.export(out_file)
        print(f"  Exported → {out_file}\n")


if __name__ == '__main__':
    process_models()
