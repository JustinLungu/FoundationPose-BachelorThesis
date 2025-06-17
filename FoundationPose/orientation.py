#!/usr/bin/env python3
"""
Compare GenAI OBJ models with GT PLY models:
1. Converts OBJ to PLY
2. Centers and scales AI models to match GT
3. Aligns using RANSAC + ICP
4. Visualizes side-by-side (GT in green, GenAI in red)
"""

import os
import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path

# CONFIGURATION
GT_FOLDER = "Limemod_preprocessed"
GEN_FOLDER = "Limemod_preprocessed"
VISUALIZE = True
NUM_SAMPLES = 20000
VOXEL_SIZE = 7.0
ANGLE_THRESHOLD = 5.0


def convert_obj_to_trimesh(obj_path):
    """Load OBJ and return a centered Trimesh object."""
    mesh = trimesh.load(obj_path, force='mesh')
    if mesh.is_empty:
        raise ValueError(f"Mesh at {obj_path} is empty.")
    mesh.vertices -= mesh.center_mass
    return mesh


def load_gt_trimesh(ply_path):
    """Load GT PLY and return a centered Trimesh object."""
    mesh = trimesh.load(ply_path, force='mesh')
    if mesh.is_empty:
        raise ValueError(f"GT mesh at {ply_path} is empty.")
    mesh.vertices -= mesh.center_mass
    return mesh


def safe_scale(mesh_gt, mesh_ai):
    """Robust scale matching based on volume, convex hull, or diagonal."""
    if mesh_gt.is_volume and mesh_ai.is_volume and mesh_gt.volume > 0 and mesh_ai.volume > 0:
        return (mesh_gt.volume / mesh_ai.volume) ** (1 / 3)
    vol_gt = mesh_gt.convex_hull.volume
    vol_ai = mesh_ai.convex_hull.volume
    if vol_gt > 0 and vol_ai > 0:
        return (vol_gt / vol_ai) ** (1 / 3)
    diag_gt = np.linalg.norm(mesh_gt.bounding_box.extents)
    diag_ai = np.linalg.norm(mesh_ai.bounding_box.extents)
    return diag_gt / diag_ai if diag_ai > 0 else 1.0


def compute_principal_axes(vertices):
    pts = vertices - vertices.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, np.argsort(eigvals)[::-1]]


def angle_between(v1, v2):
    cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(abs(cosv), -1.0, 1.0)))


def sample_point_cloud(mesh, num=NUM_SAMPLES):
    pts = np.array(mesh.sample(num))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30))
    return pcd


def align_with_icp(mesh_gt, mesh_ai):
    """Align mesh_ai to mesh_gt using RANSAC + ICP."""
    pcd_gt = sample_point_cloud(mesh_gt)
    pcd_ai = sample_point_cloud(mesh_ai)

    fpfh_gt = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_gt, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=100))
    fpfh_ai = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_ai, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=100))

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_ai, pcd_gt, fpfh_ai, fpfh_gt, True,
        VOXEL_SIZE * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOXEL_SIZE * 1.5)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

    mesh_ai.apply_transform(result_ransac.transformation)

    result_icp = o3d.pipelines.registration.registration_icp(
        sample_point_cloud(mesh_ai, 10000), sample_point_cloud(mesh_gt, 10000),
        VOXEL_SIZE, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    mesh_ai.apply_transform(result_icp.transformation)
    # print("ICP transform result:\n", result_icp.transformation)


def trimesh_to_o3d(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def compare_models(gt_folder, gen_folder):
    gt_files = list(Path(gt_folder).glob('*.ply'))
    print(f"\n{'Model':<15}{'Orig_d':>8}{'Gen_d':>8}{'Axis1':>8}{'Axis2':>8}{'Axis3':>8}  Status")
    print('-' * 64)

    for gt_file in gt_files:
        obj_file = Path(gen_folder) / (gt_file.stem + '.obj')
        if not obj_file.exists():
            print(f"{gt_file.stem:<15} MISSING OBJ")
            continue

        try:
            mesh_gt = load_gt_trimesh(gt_file)
            mesh_ai = convert_obj_to_trimesh(obj_file)
        except Exception as e:
            print(f"{gt_file.stem:<15} ERROR: {e}")
            continue

        scale_factor = safe_scale(mesh_gt, mesh_ai)
        mesh_ai.apply_scale(scale_factor)

        align_with_icp(mesh_gt, mesh_ai)

        # Compute axes and deviations
        axes_gt = compute_principal_axes(mesh_gt.vertices)
        axes_ai = compute_principal_axes(mesh_ai.vertices)
        angles = [angle_between(axes_gt[:, i], axes_ai[:, i]) for i in range(3)]
        d_gt = np.linalg.norm(mesh_gt.bounding_box.extents)
        d_ai = np.linalg.norm(mesh_ai.bounding_box.extents)
        status = 'OK' if all(a <= ANGLE_THRESHOLD for a in angles) else 'MISALIGNED'

        print(f"{gt_file.stem:<15}{d_gt:8.2f}{d_ai:8.2f}{angles[0]:8.2f}{angles[1]:8.2f}{angles[2]:8.2f}  {status}")

        if VISUALIZE:
            o3d_gt = trimesh_to_o3d(mesh_gt)
            o3d_ai = trimesh_to_o3d(mesh_ai)

            o3d_gt.paint_uniform_color([0, 1, 0])  # Green
            o3d_ai.paint_uniform_color([1, 0, 0])  # Red

            # Translate AI model to show side by side
            extent = o3d_gt.get_axis_aligned_bounding_box().get_extent()[0]
            o3d_ai.translate([extent * 1.5, 0, 0])

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries(
                [o3d_gt, o3d_ai, frame],
                window_name=f"Comparison: {gt_file.stem}",
                width=1024,
                height=768
            )

        test_export_path = Path(gen_folder) / f"{gt_file.stem}.ply"
        mesh_ai.export(test_export_path)
        print(f"Saved: {test_export_path}")


if __name__ == "__main__":
    os.makedirs(GT_FOLDER, exist_ok=True)
    os.makedirs(GEN_FOLDER, exist_ok=True)
    compare_models(GT_FOLDER, GEN_FOLDER)
