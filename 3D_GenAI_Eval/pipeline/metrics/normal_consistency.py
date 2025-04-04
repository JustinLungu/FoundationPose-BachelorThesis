import numpy as np
import trimesh
import open3d as o3d
import os
import matplotlib.pyplot as plt
from ..constants import NORMAL_CONSISTENCY_THRESHOLDS
from .base import BaseMetric


class NormalConsistencyEvaluator(BaseMetric):
    def __init__(self, mesh_gt: trimesh.Trimesh, mesh_ai: trimesh.Trimesh, model_dir: str):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai
        self.model_dir = model_dir

    def _sample_and_estimate_normals(self, mesh, num_points=5000):
        pts = np.array(mesh.sample(num_points))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
        )
        normals = np.asarray(pcd.normals)
        return pts, normals

    def compute(self, visualize=False):
        pts_gt, normals_gt = self._sample_and_estimate_normals(self.mesh_gt)
        pts_ai, normals_ai = self._sample_and_estimate_normals(self.mesh_ai)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(pts_gt)

        pcd_tree = o3d.geometry.KDTreeFlann(pcd_gt)

        matched_normals = []
        for i in range(len(pts_ai)):
            _, idx, _ = pcd_tree.search_knn_vector_3d(pts_ai[i], 1)
            matched_normals.append(normals_gt[idx[0]])

        matched_normals = np.array(matched_normals)

        # Flip inconsistent normals
        dot_products = np.sum(normals_ai * matched_normals, axis=1)
        flipped = dot_products < 0
        normals_ai[flipped] *= -1
        dot_products = np.sum(normals_ai * matched_normals, axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)

        angles = np.arccos(dot_products)
        angle_degrees = np.degrees(angles)

        normal_consistency = 1.0 - np.mean(angle_degrees) / 180.0

        if visualize:
            self._visualize_angle_map(pts_ai, angle_degrees)

        return normal_consistency

    def _visualize_angle_map(self, pts_ai, angle_degrees):
        # 3D Scatter plot of angle errors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        vmin = 0
        vmax = int(np.ceil(min(180, np.max(angle_degrees))))

        mean_angle = np.mean(angle_degrees)
        max_angle = np.max(angle_degrees)

        sc = ax.scatter(
            pts_ai[:, 0], pts_ai[:, 1], pts_ai[:, 2],
            c=angle_degrees, cmap='RdYlGn_r', s=1,
            vmin=vmin, vmax=vmax
        )

        ax.set_title(f"Normal Consistency Angle Map (0°–180°)\nMean: {mean_angle:.2f}°, Max: {max_angle:.2f}°")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Angle Difference (°)")

        plt.tight_layout()
        scatter_path = os.path.join(self.model_dir, "normal_consistency_vis.png")
        plt.savefig(scatter_path)
        plt.close()

        # Histogram of angle differences
        plt.figure()
        plt.hist(angle_degrees, bins=90, range=(0, 180), color='skyblue', edgecolor='black')
        plt.xlabel("Angle Difference (°)")
        plt.ylabel("Frequency")
        plt.title("Normal Angle Distribution (0°–180°)")
        hist_path = os.path.join(self.model_dir, "normal_angle_histogram.png")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()


    def get_class(self, score):
        return super().get_class(score, NORMAL_CONSISTENCY_THRESHOLDS, reverse=False)
