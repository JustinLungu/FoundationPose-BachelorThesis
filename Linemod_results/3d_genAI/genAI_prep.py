#!/usr/bin/env python3
"""
Convert GenAI OBJ models to PLY, scale them to match the original models,
robustly align them using RANSAC and ICP, save the final aligned models to genAI_ply,
and optionally visualize and print detailed alignment information similar to verify_alignment.py.

Assumes:
  - genAI_models/<category>/model.obj
  - original_models/obj_XX.ply

Outputs:
  - genAI_ply/obj_XX.ply (scaled and aligned)
  - Detailed printed alignment information (bounding-box diameters and axis deviations)
  - Optional visualization
"""
import os
import sys
import trimesh
import numpy as np
import open3d as o3d

CATEGORY_ID_MAP = {
    'camera':     4,
    'cat':        6,
    'drill':      8,
    'duck':       9,
    'egg_carton': 10,
    'gorilla':    1,
}

VISUALIZE = True
ANGLE_THRESHOLD = 5.0

def safe_scale(mesh_gt, mesh_ai):
    if mesh_gt.is_volume and mesh_ai.is_volume and mesh_gt.volume > 0 and mesh_ai.volume > 0:
        return (mesh_gt.volume / mesh_ai.volume) ** (1/3)
    vol_gt, vol_ai = mesh_gt.convex_hull.volume, mesh_ai.convex_hull.volume
    if vol_gt > 0 and vol_ai > 0:
        return (vol_gt / vol_ai) ** (1/3)
    diag_gt = np.linalg.norm(mesh_gt.bounding_box.extents)
    diag_ai = np.linalg.norm(mesh_ai.bounding_box.extents)
    return diag_gt / diag_ai if diag_ai > 0 else 1.0

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

def sample_point_cloud(mesh, num_samples=5000):
    points = np.array(mesh.sample(num_samples))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    return pcd

def align_meshes(mesh_gt, mesh_ai):
    voxel_size = 5.0
    pcd_gt = sample_point_cloud(mesh_gt)
    pcd_ai = sample_point_cloud(mesh_ai)

    fpfh_gt = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_gt, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    fpfh_ai = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_ai, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_ai, pcd_gt, fpfh_ai, fpfh_gt, True,
        voxel_size*1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size*1.5)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    mesh_ai.apply_transform(result_ransac.transformation)

    result_icp = o3d.pipelines.registration.registration_icp(
        sample_point_cloud(mesh_ai, 10000), sample_point_cloud(mesh_gt, 10000),
        5.0, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    mesh_ai.apply_transform(result_icp.transformation)

def visualize_pair(mesh_o, mesh_g, title):
    mesh_o.visual.vertex_colors = [255, 0, 0, 100]
    mesh_g.visual.vertex_colors = [0, 255, 0, 100]
    scene = trimesh.Scene([mesh_o, mesh_g])
    scene.show(title=title)

import shutil

def process_and_align():
    base = os.path.dirname(os.path.realpath(__file__))
    src, dst = os.path.join(base, 'genAI_models'), os.path.join(base, 'genAI_ply')
    orig_root = os.path.join(base, 'original_models')

    os.makedirs(dst, exist_ok=True)
    models_info_file = os.path.join(orig_root, 'models_info.yml')
    if os.path.isfile(models_info_file):
        shutil.copy(models_info_file, os.path.join(dst, 'models_info.yml'))

    if not os.path.isdir(src) or not os.path.isdir(orig_root):
        print("Source or original models directory missing.")
        sys.exit(1)

    header = f"{'Model':<10}{'Orig_d':>8}{'Gen_d':>8}{'Axis1':>8}{'Axis2':>8}{'Axis3':>8}  Status"
    print(header)
    print('-' * len(header))

    for category in sorted(os.listdir(src)):
        obj_file = os.path.join(src, category, 'model.obj')
        obj_id = CATEGORY_ID_MAP.get(category)

        if obj_id is None or not os.path.isfile(obj_file):
            continue

        gt_ply = os.path.join(orig_root, f'obj_{obj_id:02d}.ply')
        if not os.path.isfile(gt_ply):
            continue

        mesh_ai = trimesh.load(obj_file, force='mesh')
        mesh_gt = trimesh.load(gt_ply, force='mesh')
        if mesh_ai.is_empty or mesh_gt.is_empty:
            continue

        mesh_ai.vertices -= mesh_ai.center_mass
        mesh_gt.vertices -= mesh_gt.center_mass

        scale_factor = safe_scale(mesh_gt, mesh_ai)
        mesh_ai.apply_scale(scale_factor)

        align_meshes(mesh_gt, mesh_ai)

        d_o = np.linalg.norm(mesh_gt.bounding_box.extents)
        d_g = np.linalg.norm(mesh_ai.bounding_box.extents)

        axes_o = compute_principal_axes(mesh_gt.vertices)
        axes_g = compute_principal_axes(mesh_ai.vertices)
        angles = [angle_between(axes_o[:, i], axes_g[:, i]) for i in range(3)]
        print(f"{category:<10}{d_o:8.2f}{d_g:8.2f}{angles[0]:8.2f}{angles[1]:8.2f}{angles[2]:8.2f}")

        out_path = os.path.join(dst, f'obj_{obj_id:02d}.ply')
        mesh_ai.export(out_path)

        if VISUALIZE:
            print(f"  Visualizing {category} (orig red, gen green)...")
            visualize_pair(mesh_gt, mesh_ai, title=f"{category} alignment")

if __name__ == '__main__':
    process_and_align()
