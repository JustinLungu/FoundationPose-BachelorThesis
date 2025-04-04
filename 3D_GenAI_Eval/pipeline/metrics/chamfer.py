import matplotlib.pyplot as plt
import numpy as np
import os
import open3d as o3d

from .base import BaseMetric
from ..constants import CHAMFER_THRESHOLDS

class ChamferMetric(BaseMetric):
    def __init__(self, mesh_gt, mesh_ai, model_dir, num_samples=5000):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai
        self.num_samples = num_samples
        self.model_dir = model_dir

    def _to_open3d_pcd(self, mesh):
        points = np.asarray(mesh.sample(self.num_samples))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd, points

    def _visualize_errors(self, pts_gt, pts_ai, gt_dists, ai_dists):
        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        sc1 = ax1.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], c=gt_dists, cmap='RdYlGn_r', s=1)
        ax1.set_title('GT → AI Distance Error')
        plt.colorbar(sc1, ax=ax1)

        ax2 = fig.add_subplot(122, projection='3d')
        sc2 = ax2.scatter(pts_ai[:, 0], pts_ai[:, 1], pts_ai[:, 2], c=ai_dists, cmap='RdYlGn_r', s=1)
        ax2.set_title('AI → GT Distance Error')
        plt.colorbar(sc2, ax=ax2)

        out_path = os.path.join(self.model_dir, "chamfer_error_vis.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def compute(self) -> float:
        _, pts_gt = self._to_open3d_pcd(self.mesh_gt)
        _, pts_ai = self._to_open3d_pcd(self.mesh_ai)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(pts_gt)

        pcd_ai = o3d.geometry.PointCloud()
        pcd_ai.points = o3d.utility.Vector3dVector(pts_ai)

        gt_tree = o3d.geometry.KDTreeFlann(pcd_gt)
        ai_tree = o3d.geometry.KDTreeFlann(pcd_ai)

        gt_dists = []
        for pt in pts_gt:
            [_, idx, _] = ai_tree.search_knn_vector_3d(pt, 1)
            nearest = np.asarray(pcd_ai.points)[idx[0]]
            gt_dists.append(np.linalg.norm(pt - nearest)**2)

        ai_dists = []
        for pt in pts_ai:
            [_, idx, _] = gt_tree.search_knn_vector_3d(pt, 1)
            nearest = np.asarray(pcd_gt.points)[idx[0]]
            ai_dists.append(np.linalg.norm(pt - nearest)**2)

        self._visualize_errors(pts_gt, pts_ai, gt_dists, ai_dists)

        chamfer_dist = np.mean(gt_dists) + np.mean(ai_dists)
        return chamfer_dist

    def get_class(self, score):
        return super().get_class(score, CHAMFER_THRESHOLDS, reverse=True)