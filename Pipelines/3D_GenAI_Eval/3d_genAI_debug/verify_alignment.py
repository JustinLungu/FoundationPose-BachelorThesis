#!/usr/bin/env python3
"""
Verify, align using RANSAC and ICP, visualize original and GenAI-generated models, and save aligned GenAI models.

Assumes:
  - 'original_models/obj_XX.ply'
  - 'genAI_ply/obj_XX.ply'

Outputs:
  - Printed diameters (original vs genAI) and axis deviations
  - Optional interactive overlay (red vs green)
  - Aligned models saved to 'aligned_genAI_ply/obj_XX.ply'
"""
import os
import sys
import numpy as np
import trimesh
import open3d as o3d

VISUALIZE = True
ANGLE_THRESHOLD = 5.0
ALIGNED_DIR = 'aligned_genAI_ply'


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
    pcd_gt = sample_point_cloud(mesh_gt)
    pcd_ai = sample_point_cloud(mesh_ai)

    voxel_size = 5.0

    fpfh_gt = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_gt, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    fpfh_ai = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_ai, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_ai, pcd_gt, fpfh_ai, fpfh_gt, True,
        voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
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


def verify_model(orig_path, gen_path, aligned_path):
    mesh_o = trimesh.load(orig_path, force='mesh')
    mesh_g = trimesh.load(gen_path, force='mesh')
    if mesh_o.is_empty or mesh_g.is_empty:
        print(f"Warning: empty mesh, skipping {os.path.basename(orig_path)}")
        return None

    align_meshes(mesh_o, mesh_g)

    axes_o = compute_principal_axes(mesh_o.vertices)
    axes_g = compute_principal_axes(mesh_g.vertices)
    angles = [angle_between(axes_o[:, i], axes_g[:, i]) for i in range(3)]

    mesh_g.export(aligned_path)

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
    gen_dir = os.path.join(script_dir, 'genAI_ply')
    aligned_dir = os.path.join(script_dir, ALIGNED_DIR)
    os.makedirs(aligned_dir, exist_ok=True)

    if not os.path.isdir(orig_dir) or not os.path.isdir(gen_dir):
        print("Error: 'original_models' and/or 'genAI_ply' directories not found.")
        sys.exit(1)

    print("Verifying, aligning, and saving model orientations:\n")
    header = f"{'Model':<10}{'Orig_d':>8}{'Gen_d':>8}{'Axis1':>8}{'Axis2':>8}{'Axis3':>8}  Status"
    print(header)
    print('-' * len(header))

    for fname in sorted(os.listdir(orig_dir)):
        if not fname.lower().endswith('.ply'):
            continue
        orig_path = os.path.join(orig_dir, fname)
        gen_path = os.path.join(gen_dir, fname)
        aligned_path = os.path.join(aligned_dir, fname)
        if not os.path.isfile(gen_path):
            print(f"{fname:<10} MISSING in genAI_ply")
            continue

        result = verify_model(orig_path, gen_path, aligned_path)
        if result is None:
            continue
        angles, mesh_o, mesh_g = result

        d_o = np.linalg.norm(mesh_o.bounding_box.extents)
        d_g = np.linalg.norm(mesh_g.bounding_box.extents)

        status = 'OK' if all(a <= ANGLE_THRESHOLD for a in angles) else 'MISALIGNED'
        print(f"{fname[:-4]:<10}{d_o:8.2f}{d_g:8.2f}{angles[0]:8.2f}{angles[1]:8.2f}{angles[2]:8.2f}  {status}")

        if VISUALIZE:
            print(f"  Visualizing {fname} (orig red, gen green)...")
            visualize_pair(mesh_o, mesh_g, title=f"{fname} alignment")


if __name__ == '__main__':
    main()
