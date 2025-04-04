import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
import os
from scipy.optimize import linear_sum_assignment

from ..constants import EMD_THRESHOLDS
from .base import BaseMetric


class EMDEvaluator(BaseMetric):
    def __init__(self, mesh_gt: trimesh.Trimesh, mesh_ai: trimesh.Trimesh, model_dir: str):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai
        self.model_dir = model_dir

    def _sample_points(self, num_points=1000):
        pts_gt = np.array(self.mesh_gt.sample(num_points))
        pts_ai = np.array(self.mesh_ai.sample(num_points))
        return pts_gt, pts_ai

    def compute(self, visualize=False):
        pts_gt, pts_ai = self._sample_points()

        # Compute pairwise distance matrix
        dists = np.linalg.norm(pts_gt[:, np.newaxis, :] - pts_ai[np.newaxis, :, :], axis=2)

        # Solve optimal transport matching
        row_ind, col_ind = linear_sum_assignment(dists)
        matched_dists = dists[row_ind, col_ind]
        emd_score = np.mean(matched_dists)

        if visualize:
            self._visualize_matches(pts_gt, pts_ai, row_ind, col_ind, matched_dists)

        return emd_score

    def _visualize_matches(self, pts_gt, pts_ai, row_ind, col_ind, dists):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot AI and GT points
        ax.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], c='red', s=3, label='GT')
        ax.scatter(pts_ai[:, 0], pts_ai[:, 1], pts_ai[:, 2], c='green', s=3, label='AI')

        # Normalize distances for coloring
        norm_dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists) + 1e-8)
        cmap = plt.get_cmap('RdYlGn_r')

        for i in range(len(row_ind)):
            p1 = pts_gt[row_ind[i]]
            p2 = pts_ai[col_ind[i]]
            color = cmap(norm_dists[i])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color, linewidth=0.5)

        ax.set_title("Earth Mover’s Distance (Matched Point Lines)")
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(self.model_dir, "emd_vis.png")
        plt.savefig(out_path)
        plt.close()


    def get_class(self, score):
        return super().get_class(score, EMD_THRESHOLDS, reverse=True)
