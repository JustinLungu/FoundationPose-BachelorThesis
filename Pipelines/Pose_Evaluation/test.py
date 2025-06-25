#!/usr/bin/env python3
"""
debug_step2_onward.py

This script assumes you have already computed M_align (the 4×4 DF→GT alignment)
and applied it to your DF mesh.  It picks up _after_ object‐space alignment was
successful, and focuses only on “Step 2→Step 5” (camera‐space visualization).

Run from your Pose_Evaluation folder like:
    python debug_step2_onward.py
"""

import os
import sys
import numpy as np
import trimesh
import open3d as o3d
import yaml

# ‒———————————————————————————————————————————————————————————————————————————
#  (ONLY EDIT THESE FILEPATHS AS NEEDED)
# ‒———————————————————————————————————————————————————————————————————————————

# (1) Already‐aligned GT and DF meshes (after M_align was applied to DF)
#     – In practice, you can just reload your original GT mesh + reapply M_align to DF
#       in this snippet.  For clarity, we’ll reload both now and re‐apply “M_align → DF_mesh.”
GT_PLY   = "3d_models/original_models/obj_01.ply"
DF_PLY   = "3d_models/dreamfusion/obj_01.ply"       # <— this DF_PLY should already be centered & scaled & ICP‐aligned

# (2) Frame‐0 YAMLs for LINEMOD GT and DF‐predictions
GT_YAML  = "reformatted/gt_obj_01.yml"
DF_YAML  = "reformatted/dreamfusion_obj_01_pred.yml"

# (3) The 4×4 object‐space alignment you computed earlier (DF→GT)
#     – If you want to hardcode it, paste it here.  Otherwise, you can recompute it by
#       re‐running your align_with_icp() function on the two meshes.  For this example,
#       we’ll assume you already have it as “M_align.npy” on disk:
M_ALIGN_PATH = "reformatted/obj01_df_to_gt_M_align.npy"

# (4) Number of points to sample from each mesh (still in millimeters)
NUM_SAMPLES = 5000

# ‒———————————————————————————————————————————————————————————————————————————
#  HELPER FUNCTIONS
# ‒———————————————————————————————————————————————————————————————————————————

def load_transform_from_yaml_first_frame(yaml_path: str) -> np.ndarray:
    """
    Loads the FIRST (lowest numeric) frame index in the given YAML, under its only object key.
    Returns a 4×4 numpy array.  Raises if no valid frame is found.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Expect exactly one top‐level key (object ID)
    if len(data) != 1:
        raise ValueError(f"[load_transform] YAML '{yaml_path}' must have exactly one top‐level object ID.")
    obj_id = list(data.keys())[0]
    frame_dict = data[obj_id]

    if not isinstance(frame_dict, dict) or len(frame_dict) == 0:
        raise ValueError(f"[load_transform] No frames found under object '{obj_id}' in '{yaml_path}'.")

    # Gather any numeric frame‐keys
    numeric_keys = []
    for k in frame_dict.keys():
        try:
            numeric_keys.append(int(k))
        except:
            pass

    if len(numeric_keys) > 0:
        chosen = min(numeric_keys)
        frame_key = chosen if chosen in frame_dict else str(chosen)
    else:
        # Fallback: just pick the lexicographically first key
        sorted_keys = sorted(frame_dict.keys(), key=lambda x: str(x))
        frame_key = sorted_keys[0]

    if frame_key not in frame_dict:
        raise ValueError(f"[load_transform] Frame key '{frame_key}' not found in '{yaml_path}'.")

    mat4_list = frame_dict[frame_key].get(obj_id, None)
    if mat4_list is None:
        raise ValueError(f"[load_transform] No 4×4 matrix under object '{obj_id}', frame '{frame_key}' in '{yaml_path}'.")

    T = np.array(mat4_list, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"[load_transform] Matrix has shape {T.shape}, expected (4×4).")

    return T


def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    Converts a trimesh.Trimesh → open3d.geometry.TriangleMesh, computing vertex normals.
    """
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


# ‒———————————————————————————————————————————————————————————————————————————
#  MAIN
# ‒———————————————————————————————————————————————————————————————————————————

def main():
    # 1) Check files exist
    for p in (GT_PLY, DF_PLY, GT_YAML, DF_YAML, M_ALIGN_PATH):
        if not os.path.exists(p):
            print(f"[ERROR] Cannot find '{p}'.")
            sys.exit(1)

    # 2) Reload the GT and DF meshes *in millimeters*, so that we can re‐apply M_align
    print("\nLoading GT mesh (mm) …")
    mesh_gt = trimesh.load(GT_PLY, force="mesh")
    if mesh_gt.is_empty:
        print("[ERROR] GT mesh is empty!"); sys.exit(1)
    mesh_gt.vertices -= mesh_gt.center_mass

    print("Loading DF mesh (mm) …")
    mesh_df = trimesh.load(DF_PLY, force="mesh")
    if mesh_df.is_empty:
        print("[ERROR] DF mesh is empty!"); sys.exit(1)
    mesh_df.vertices -= mesh_df.center_mass

    # 3) Load M_align (DF→GT) that you saved from your previous object‐space run
    print("Loading M_align …")
    M_align = np.load(M_ALIGN_PATH)
    if M_align.shape != (4, 4):
        print(f"[ERROR] M_align.npy has shape {M_align.shape}, but expected (4×4)."); sys.exit(1)

    # 4) Apply M_align to DF mesh (so it overlaps GT in object‐space)
    print("Applying DF→GT alignment (object‐space) …")
    mesh_df.apply_transform(M_align)

    # 5) Sample each mesh (still in mm) → convert to meters (divide by 1000)
    print("Sampling GT and DF point‐clouds …")
    o3d_gt_mesh = trimesh_to_o3d(mesh_gt)
    o3d_df_mesh = trimesh_to_o3d(mesh_df)

    pcd_gt = o3d_gt_mesh.sample_points_uniformly(number_of_points=NUM_SAMPLES)
    pcd_df = o3d_df_mesh.sample_points_uniformly(number_of_points=NUM_SAMPLES)

    # Convert both to meters
    pts_gt = np.asarray(pcd_gt.points) / 1000.0
    pts_df = np.asarray(pcd_df.points) / 1000.0
    pcd_gt.points = o3d.utility.Vector3dVector(pts_gt)
    pcd_df.points = o3d.utility.Vector3dVector(pts_df)

    # 6) Load frame‐0 camera→object transforms from YAML
    print("Loading frame‐0 camera→object transforms …")
    T_gt0 = load_transform_from_yaml_first_frame(GT_YAML)
    T_df0 = load_transform_from_yaml_first_frame(DF_YAML)

    print(f"\nT_gt0 (GT) =\n{T_gt0}\n")
    print(f"T_df0 (DF, raw) =\n{T_df0}\n")

    # 7) Bake alignment into DF’s camera→object transform
    ##    If M_align maps DF_object → GT_object in object‐space, then to “lift”
    ##    DF→camera, we must do:  T_df0_aligned = T_df0 @ inv(M_align).
    T_df0_aligned = T_df0 @ np.linalg.inv(M_align)
    print(f"T_df0_aligned (DF_in_camera_after_baking_icp) =\n{T_df0_aligned}\n")

    # 8) Transform both point‐clouds into camera‐space
    print("Transforming both clouds into camera‐space …")
    pcd_gt.transform(T_gt0)
    pcd_df.transform(T_df0_aligned)

    # 9) Color & display
    pcd_gt.paint_uniform_color([0.0, 1.0, 0.0])   # green
    pcd_df.paint_uniform_color([1.0, 0.0, 0.0])   # red

    print("Displaying CAMERA‐SPACE overlay (GT=green vs DF=red)…")
    o3d.visualization.draw_geometries(
        [pcd_gt, pcd_df],
        window_name="Camera‐space Overlay: GT vs DF",
        width=800,
        height=600,
    )

    print("Done.\n")


if __name__ == "__main__":
    main()
