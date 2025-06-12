#!/usr/bin/env python3
"""
debug_align_and_camera_frame0_with_pose_check.py

1) Load GT PLY & DF OBJ → center both.
2) Compute “safe” scale (DF→GT) → scale DF.
3) RANSAC + ICP to align DF→GT in object space.
4) Display object-space overlay (green=GT; red=DF).
5) Load frame-0 camera→object transforms (GT & DF) from YAML (in meters).
6) Bake in ICP by doing T_df0_aligned = M_align @ T_df0.
2.5) Compute rotation‐ and translation‐error between T_gt0 and T_df0_aligned.
7) Sample both meshes (in meters).
8) Invert each camera→object matrix and apply to points → camera-space.
9) Display final camera-space overlay.
"""

import os
import sys
import numpy as np
import trimesh
import open3d as o3d
import yaml

NUM_SAMPLES   = 5000
VOXEL_SIZE    = 5.0

GT_PLY        = "3d_models/original_models/obj_01.ply"
DF_OBJ        = "3d_models/dreamfusion/obj_01.ply"
GT_YAML       = "reformatted/gt_obj_01.yml"
DF_YAML       = "reformatted/dreamfusion_obj_01_pred.yml"

def load_gt_trimesh(ply_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(ply_path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"[load_gt_trimesh] {ply_path} is empty!")
    mesh.vertices -= mesh.center_mass
    return mesh

def convert_obj_to_trimesh(obj_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"[convert_obj_to_trimesh] {obj_path} is empty!")
    mesh.vertices -= mesh.center_mass
    return mesh

def safe_scale(mesh_gt: trimesh.Trimesh, mesh_df: trimesh.Trimesh) -> float:
    if mesh_gt.is_volume and mesh_df.is_volume and mesh_gt.volume > 0 and mesh_df.volume > 0:
        return (mesh_gt.volume / mesh_df.volume) ** (1.0 / 3.0)
    vol_gt = mesh_gt.convex_hull.volume
    vol_df = mesh_df.convex_hull.volume
    if vol_gt > 0 and vol_df > 0:
        return (vol_gt / vol_df) ** (1.0 / 3.0)
    diag_gt = np.linalg.norm(mesh_gt.bounding_box.extents)
    diag_df = np.linalg.norm(mesh_df.bounding_box.extents)
    return (diag_gt / diag_df) if (diag_df > 0) else 1.0

def sample_point_cloud(mesh: trimesh.Trimesh, num: int) -> o3d.geometry.PointCloud:
    pts = np.array(mesh.sample(num))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30)
    )
    return pcd

def align_with_icp(mesh_gt: trimesh.Trimesh, mesh_df: trimesh.Trimesh) -> np.ndarray:
    pcd_gt = sample_point_cloud(mesh_gt, NUM_SAMPLES)
    pcd_df = sample_point_cloud(mesh_df, NUM_SAMPLES)

    fpfh_gt = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_gt,
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=100),
    )
    fpfh_df = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_df,
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=100),
    )

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_df,
        pcd_gt,
        fpfh_df,
        fpfh_gt,
        True,
        VOXEL_SIZE * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOXEL_SIZE * 1.5),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )
    T_ransac = result_ransac.transformation
    mesh_df.apply_transform(T_ransac)

    pcd_gt2 = sample_point_cloud(mesh_gt, NUM_SAMPLES * 2)
    pcd_df2 = sample_point_cloud(mesh_df, NUM_SAMPLES * 2)
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd_df2,
        pcd_gt2,
        VOXEL_SIZE,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    T_icp = result_icp.transformation
    mesh_df.apply_transform(T_icp)

    M_align = T_icp @ T_ransac
    return M_align

def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def load_first_frame_from_yaml(yaml_path: str) -> (np.ndarray, int):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if len(data.keys()) != 1:
        raise ValueError(f"[load_first_frame] '{yaml_path}' must have exactly one top‐level object ID.")
    obj_id = list(data.keys())[0]
    frame_dict = data[obj_id]
    if not isinstance(frame_dict, dict) or len(frame_dict) == 0:
        raise ValueError(f"[load_first_frame] No frames under '{obj_id}' in '{yaml_path}'.")

    numeric_keys = []
    for k in frame_dict.keys():
        try:
            numeric_keys.append(int(k))
        except:
            pass

    if len(numeric_keys) > 0:
        chosen_frame = min(numeric_keys)
        chosen_key = chosen_frame if (chosen_frame in frame_dict) else str(chosen_frame)
    else:
        sorted_keys = sorted(frame_dict.keys(), key=lambda x: str(x))
        chosen_key = sorted_keys[0]
        try:
            chosen_frame = int(chosen_key)
        except:
            chosen_frame = -1

    if chosen_key not in frame_dict:
        raise ValueError(f"[load_first_frame] Key '{chosen_key}' not in '{yaml_path}'.")

    mat4_list = frame_dict[chosen_key].get(obj_id, None)
    if mat4_list is None:
        raise ValueError(f"[load_first_frame] No entry for '{obj_id}' under frame '{chosen_key}' in '{yaml_path}'.")

    T = np.array(mat4_list, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"[load_first_frame] Matrix shape {T.shape}, expected (4×4).")
    return T, chosen_frame

def main():
    # 1) Ensure files exist
    for p in (GT_PLY, DF_OBJ, GT_YAML, DF_YAML):
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    # 2) Stage 1: Load and center both meshes (in meters)
    print("\nStage 1: Loading and centering meshes …")
    mesh_gt = load_gt_trimesh(GT_PLY)
    mesh_df = convert_obj_to_trimesh(DF_OBJ)

    # 3) Scale DF→GT
    scale_factor = safe_scale(mesh_gt, mesh_df)
    print(f"  Scale factor (DF→GT) = {scale_factor:.6f}")
    mesh_df.apply_scale(scale_factor)

    # 4) RANSAC + ICP → align DF→GT in object space
    print("  Running RANSAC + ICP …")
    M_align = align_with_icp(mesh_gt, mesh_df)
    print("  … Alignment complete in object space.\n")

    # 5) Display object‐space overlay
    print("Displaying **object‐space** overlay (post‐ICP)…")
    o3d_gt_obj = trimesh_to_o3d(mesh_gt)
    o3d_df_obj = trimesh_to_o3d(mesh_df)
    o3d_gt_obj.paint_uniform_color([0.0, 1.0, 0.0])  # green
    o3d_df_obj.paint_uniform_color([1.0, 0.0, 0.0])  # red
    o3d.visualization.draw_geometries(
        [o3d_gt_obj, o3d_df_obj],
        window_name="Object‐space Overlay: GT (green) vs DF (red)",
        width=800,
        height=600,
    )
    print("If the above looks properly overlapped, press ENTER to continue…")
    input()

    # 6) Stage 2: Load frame‐0 camera→object poses from YAML (in meters)
    print("\nStage 2: Loading frame‐0 camera→object transforms …")
    T_gt0, frame_gt = load_first_frame_from_yaml(GT_YAML)     # already in meters
    T_df0, frame_df = load_first_frame_from_yaml(DF_YAML)     # already in meters
    print(f"  GT: frame = {frame_gt},   T_gt0 (meters) =\n{T_gt0}\n")
    print(f"  DF: frame = {frame_df},   T_df0 (meters) =\n{T_df0}\n")

    # 6.5) Bake ICP into DF’s camera→object:  T_df0_aligned = M_align @ T_df0
    T_df0_aligned = M_align @ T_df0
    print(f"  DF: T_df0_aligned (meters) =\n{T_df0_aligned}\n")

    # --- Step 2.5: Compute “pose error” between GT and DF_aligned for frame0 ---
    R_gt = T_gt0[:3, :3]
    R_df = T_df0_aligned[:3, :3]
    R_diff = R_gt.T @ R_df
    trace_val = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle_rad = np.arccos((trace_val - 1.0) / 2.0)
    angle_deg = np.degrees(angle_rad)

    t_gt = T_gt0[:3, 3]
    t_df = T_df0_aligned[:3, 3]
    trans_error = np.linalg.norm(t_gt - t_df)

    print("  → Frame 0 Pose Differences:")
    print(f"     ‣ Rotation‐error:    {angle_deg:.4f} degrees")
    print(f"     ‣ Translation‐error: {trans_error:.6f} meters\n")

    # 7) Sample both meshes (in meters)
    o3d_gt_mesh = trimesh_to_o3d(mesh_gt)
    o3d_df_mesh = trimesh_to_o3d(mesh_df)

    print("Sampling point clouds …")
    pcd_gt0 = o3d_gt_mesh.sample_points_uniformly(NUM_SAMPLES)
    pcd_df0 = o3d_df_mesh.sample_points_uniformly(NUM_SAMPLES)

    # 8) Transform into camera space by inverting camera→object
    pcd_gt0.transform(np.linalg.inv(T_gt0))
    pcd_df0.transform(np.linalg.inv(T_df0_aligned))

    pcd_gt0.paint_uniform_color([0.0, 1.0, 0.0])  # green
    pcd_df0.paint_uniform_color([1.0, 0.0, 0.0])  # red

    # 9) Display the camera‐space overlay
    print("Displaying **camera‐space** overlay …")
    o3d.visualization.draw_geometries(
        [pcd_gt0, pcd_df0],
        window_name="Camera‐space Overlay: GT (green) vs DF (red)",
        width=800,
        height=600,
    )
    print("\nDone.\n")

if __name__ == "__main__":
    main()
