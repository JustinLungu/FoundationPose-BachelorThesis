"""
Precision alignment module using RANSAC and ICP.

Pipeline:
1. RANSAC: Coarse global alignment using FPFH features
2. Multiscale ICP:
   - Stage 1: Coarse (20mm threshold)
   - Stage 2: Fine (5mm threshold)
   
Note: Requires pre-normalized meshes from Preprocessor.
"""

import numpy as np
import open3d as o3d
from pipeline.config import DEFAULT_NUM_SAMPLES, VOXEL_SIZE
np.random.seed(42)

class MeshRefiner:
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai

    def _sample_point_cloud(self, mesh, num_samples=DEFAULT_NUM_SAMPLES):
        """Convert mesh to point cloud with estimated normals.
        Used for feature-based registration.
        """
        points = np.array(mesh.sample(num_samples))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30))
        return pcd

    def apply_ransac_icp(self):
        pcd_gt = self._sample_point_cloud(self.mesh_gt)
        pcd_ai = self._sample_point_cloud(self.mesh_ai)

        def compute_fpfh(pcd):
            return o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=100)
            )

        fpfh_gt = compute_fpfh(pcd_gt)
        fpfh_ai = compute_fpfh(pcd_ai)

        distance_threshold = VOXEL_SIZE * 1.5

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_ai, pcd_gt, fpfh_ai, fpfh_gt, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        if result_ransac.fitness > 0.1:
            self.mesh_ai.apply_transform(result_ransac.transformation)
            print("[RANSAC] Applied global registration.")
        else:
            print(f"[RANSAC] Low fitness ({result_ransac.fitness:.3f}), skipping.")


        current_trans = np.eye(4)
        for thresh in (VOXEL_SIZE*3, VOXEL_SIZE, VOXEL_SIZE*0.2):
            pcd_ai = self._sample_point_cloud(self.mesh_ai, 10000)
            pcd_gt = self._sample_point_cloud(self.mesh_gt, 10000)
            result_icp = o3d.pipelines.registration.registration_icp(
                pcd_ai, pcd_gt,
                thresh,
                current_trans,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            current_trans = result_icp.transformation
            print(f"[ICP] Stage thresh={thresh:.1f}, fitness={result_icp.fitness:.3f}")
        # apply the final refined transform
        self.mesh_ai.apply_transform(current_trans)
        print("[ICP] Multistage refinement complete.")
        return self.mesh_ai