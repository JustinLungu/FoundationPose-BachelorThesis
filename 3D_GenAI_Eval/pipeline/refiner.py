import numpy as np
import open3d as o3d

class MeshRefiner:
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai

    def _sample_point_cloud(self, mesh, num_samples=5000):
        points = np.array(mesh.sample(num_samples))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
        return pcd

    def apply_ransac(self, voxel_size=5.0):
        pcd_gt = self._sample_point_cloud(self.mesh_gt)
        pcd_ai = self._sample_point_cloud(self.mesh_ai)

        def compute_fpfh(pcd):
            return o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
            )

        fpfh_gt = compute_fpfh(pcd_gt)
        fpfh_ai = compute_fpfh(pcd_ai)

        distance_threshold = voxel_size * 1.5

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
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

        self.mesh_ai.apply_transform(result.transformation)
        print("[RANSAC] Applied global registration.")

    def apply_multiscale_icp(self, coarse_thresh=20.0, fine_thresh=5.0):
        pcd_gt = self._sample_point_cloud(self.mesh_gt, num_samples=10000)
        pcd_ai = self._sample_point_cloud(self.mesh_ai, num_samples=10000)

        result_coarse = o3d.pipelines.registration.registration_icp(
            pcd_ai, pcd_gt, coarse_thresh, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        result_fine = o3d.pipelines.registration.registration_icp(
            pcd_ai, pcd_gt, fine_thresh, result_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        self.mesh_ai.apply_transform(result_fine.transformation)
        print("[ICP] Applied multiscale refinement.")