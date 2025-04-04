import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from .base import BaseMetric
from constants import HAUSDORFF_THRESHOLDS


class HausdorffDistanceEvaluator(BaseMetric):
    def __init__(self, mesh_gt, mesh_ai, model_dir, num_samples=5000):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai
        self.model_dir = model_dir
        self.num_samples = num_samples

    def compute(self):
        # Sample points
        pts_gt = np.array(self.mesh_gt.sample(self.num_samples))
        pts_ai = np.array(self.mesh_ai.sample(self.num_samples))

        # Nearest neighbor distances
        tree_gt = cKDTree(pts_gt)
        tree_ai = cKDTree(pts_ai)

        dists_gt_to_ai, _ = tree_ai.query(pts_gt, k=1)
        dists_ai_to_gt, _ = tree_gt.query(pts_ai, k=1)

        hd_gt_to_ai = np.max(dists_gt_to_ai)
        hd_ai_to_gt = np.max(dists_ai_to_gt)
        hausdorff = max(hd_gt_to_ai, hd_ai_to_gt)

        self._visualize_errors(pts_gt, pts_ai, dists_gt_to_ai, dists_ai_to_gt)
        return hausdorff

    def _visualize_errors(self, pts_gt, pts_ai, dists_gt_to_ai, dists_ai_to_gt):
        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        sc1 = ax1.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], c=dists_gt_to_ai, cmap='RdYlGn_r', s=1)
        ax1.set_title('GT → AI Distance Error')
        plt.colorbar(sc1, ax=ax1)

        ax2 = fig.add_subplot(122, projection='3d')
        sc2 = ax2.scatter(pts_ai[:, 0], pts_ai[:, 1], pts_ai[:, 2], c=dists_ai_to_gt, cmap='RdYlGn_r', s=1)
        ax2.set_title('AI → GT Distance Error')
        plt.colorbar(sc2, ax=ax2)

        out_path = os.path.join(self.model_dir, "hausdorff_error_vis.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def get_class(self, score):
        return super().get_class(score, HAUSDORFF_THRESHOLDS, reverse=True)