import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
import os

from ..constants import MEAN_CURVATURE_THRESHOLDS
from .base import BaseMetric


class MeanCurvatureEvaluator(BaseMetric):
    def __init__(self, mesh_gt: trimesh.Trimesh, mesh_ai: trimesh.Trimesh, model_dir: str):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai
        self.model_dir = model_dir

    def _sample_and_estimate_curvature(self, mesh, num_points=5000):
        pts = np.array(mesh.sample(num_points))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        curvatures = []
        for i in range(len(pts)):
            _, idx, _ = pcd_tree.search_knn_vector_3d(pts[i], 30)
            neighbors = np.asarray(pcd.points)[idx]
            cov = np.cov(neighbors.T)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.sort(eigvals)
            curvature = eigvals[0] / np.sum(eigvals)
            curvatures.append(curvature)

        return pts, np.array(curvatures)

    def compute(self, visualize=False):
        pts_gt, curv_gt = self._sample_and_estimate_curvature(self.mesh_gt)
        pts_ai, curv_ai = self._sample_and_estimate_curvature(self.mesh_ai)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(pts_gt)
        kdtree_gt = o3d.geometry.KDTreeFlann(pcd_gt)

        matched_curvs = []
        for i in range(len(pts_ai)):
            _, idx, _ = kdtree_gt.search_knn_vector_3d(pts_ai[i], 1)
            matched_curvs.append(curv_gt[idx[0]])

        matched_curvs = np.array(matched_curvs)
        curvature_errors = np.abs(curv_ai - matched_curvs)
        mean_error = np.mean(curvature_errors)

        if visualize:
            self._visualize_curvature_error(pts_ai, curvature_errors)

        return mean_error

    def _visualize_curvature_error(self, pts, errors):
        vmax_val = int(np.ceil(max(0.05, np.max(errors)) * 100)) / 100  # cap to make visually clear

        # Scatter
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=errors, cmap='RdYlGn_r', s=1, vmin=0, vmax=vmax_val)
        ax.set_title("Mean Curvature Error Map")
        plt.colorbar(sc, ax=ax)
        plt.tight_layout()
        out_path = os.path.join(self.model_dir, "mean_curvature_vis.png")
        plt.savefig(out_path)
        plt.close()

        # Histogram
        plt.figure()
        plt.hist(errors, bins=50, range=(0, 0.1), color='lightcoral', edgecolor='black')
        plt.xlabel("Curvature Difference")
        plt.ylabel("Frequency")
        plt.title("Curvature Error Distribution")
        hist_path = os.path.join(self.model_dir, "mean_curvature_histogram.png")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

    def get_class(self, score):
        return super().get_class(score, MEAN_CURVATURE_THRESHOLDS, reverse=True)
