#!/usr/bin/env python3
"""
Compare GenAI models with GT models:
1. Loads OBJ or PLY
2. Centers and scales AI to match GT
3. Aligns using RANSAC + multistage ICP (seeded)
4. Visualizes side-by-side
"""

import os
import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path

# CONFIGURATION
GT_FOLDER = "Linemod_preprocessed/original"
GEN_FOLDER = "Linemod_preprocessed/testing/dreamfusion"
VISUALIZE = True
ANGLE_THRESHOLD = 5.0
NUM_SAMPLES = 20000
VOXEL_SIZE = 7.0

# Seed for deterministic sampling
np.random.seed(42)

class MeshRefiner:
    """
    Precision alignment: RANSAC global + multistage ICP
    """
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai

    def _sample_point_cloud(self, mesh, num_samples=NUM_SAMPLES):
        pts = np.array(mesh.sample(num_samples))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30)
        )
        return pcd

    def apply_ransac_icp(self):
        def compute_fpfh(pcd):
            return o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=100)
            )

        # Global RANSAC
        pcd_gt = self._sample_point_cloud(self.mesh_gt)
        pcd_ai = self._sample_point_cloud(self.mesh_ai)
        fpfh_gt = compute_fpfh(pcd_gt)
        fpfh_ai = compute_fpfh(pcd_ai)
        thresh = VOXEL_SIZE * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_ai, pcd_gt, fpfh_ai, fpfh_gt, True,
            thresh,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thresh)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )
        if result.fitness > 0.1:
            self.mesh_ai.apply_transform(result.transformation)

        # Multistage ICP: coarse, medium, fine
        current = np.eye(4)
        for t in (VOXEL_SIZE*3, VOXEL_SIZE, VOXEL_SIZE*0.2):
            src = self._sample_point_cloud(self.mesh_ai)
            tgt = self._sample_point_cloud(self.mesh_gt)
            res = o3d.pipelines.registration.registration_icp(
                src, tgt, t, current,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            current = res.transformation
        self.mesh_ai.apply_transform(current)
        return self.mesh_ai


def load_mesh(path: Path):
    """Load OBJ or PLY and center at origin"""
    mesh = trimesh.load(path, force='mesh')
    if mesh.is_empty:
        raise ValueError(f"Mesh at {path} is empty.")
    mesh.vertices -= mesh.center_mass
    return mesh


def safe_scale(mesh_gt, mesh_ai):
    if mesh_gt.is_volume and mesh_ai.is_volume and mesh_gt.volume > 0 and mesh_ai.volume > 0:
        return (mesh_gt.volume / mesh_ai.volume) ** (1/3)
    vg = mesh_gt.convex_hull.volume
    va = mesh_ai.convex_hull.volume
    if vg > 0 and va > 0:
        return (vg / va) ** (1/3)
    dg = np.linalg.norm(mesh_gt.bounding_box.extents)
    da = np.linalg.norm(mesh_ai.bounding_box.extents)
    return dg/da if da>0 else 1.0


def compute_principal_axes(vertices):
    pts = vertices - vertices.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, np.argsort(vals)[::-1]]


def angle_between(v1, v2):
    c = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(abs(c),-1,1)))


def trimesh_to_o3d(mesh):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    m.triangles = o3d.utility.Vector3iVector(mesh.faces)
    m.compute_vertex_normals()
    return m


def compare_models(gt_folder, gen_folder):
    files = list(Path(gt_folder).glob('*.ply'))
    print(f"\n{'Model':<15}{'O_d':>6}{'G_d':>6}{'A1':>6}{'A2':>6}{'A3':>6}  Status")
    print('-'*50)
    for gt in files:
        stem = gt.stem
        # find AI mesh: .obj or .ply
        ai_obj = Path(gen_folder)/(stem+'.obj')
        ai_ply = Path(gen_folder)/(stem+'.ply')
        if ai_obj.exists(): ai_path = ai_obj
        elif ai_ply.exists(): ai_path = ai_ply
        else:
            print(f"{stem:<15} MISSING AI")
            continue

        try:
            m_gt = load_mesh(gt)
            m_ai = load_mesh(ai_path)
        except Exception as e:
            print(f"{stem:<15} ERROR: {e}")
            continue

        # match AI centroid to GT centroid
        gt_c = m_gt.center_mass
        ai_c = m_ai.center_mass
        m_ai.vertices += (gt_c - ai_c)

        # scale AI to GT via volume
        s = safe_scale(m_gt, m_ai)
        m_ai.apply_scale(s)

        # align
        ref = MeshRefiner(m_gt, m_ai)
        m_ai = ref.apply_ransac_icp()

        agt = compute_principal_axes(m_gt.vertices)
        aai = compute_principal_axes(m_ai.vertices)
        ang = [angle_between(agt[:,i],aai[:,i]) for i in range(3)]
        dgt = np.linalg.norm(m_gt.bounding_box.extents)
        dai = np.linalg.norm(m_ai.bounding_box.extents)
        ok = all(a<=ANGLE_THRESHOLD for a in ang)
        stat = 'OK' if ok else 'MIS'
        print(f"{stem:<15}{dgt:6.1f}{dai:6.1f}{ang[0]:6.1f}{ang[1]:6.1f}{ang[2]:6.1f}  {stat}")

        if VISUALIZE:
            g3 = trimesh_to_o3d(m_gt); a3 = trimesh_to_o3d(m_ai)
            g3.paint_uniform_color([0,1,0]); a3.paint_uniform_color([1,0,0])
            dist = g3.get_axis_aligned_bounding_box().get_extent()[0]
            a3.translate([dist*1.5,0,0])
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([g3,a3,frame],window_name=stem,width=800,height=600)

        out = Path(gen_folder)/f"{stem}.ply"
        m_ai.export(out)
        print(f"Saved {out}")

if __name__ == '__main__':
    os.makedirs(GT_FOLDER,exist_ok=True)
    os.makedirs(GEN_FOLDER,exist_ok=True)
    compare_models(GT_FOLDER,GEN_FOLDER)
