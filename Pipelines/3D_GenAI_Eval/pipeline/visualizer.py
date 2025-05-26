"""
3D comparison visualization generator.

Features:
- Point cloud sampling for lightweight rendering
- Consistent color coding (Red=GT, Green=AI)
- Optional interactive view or static image export
- Fixed viewpoint for comparable visualizations

Note: Uses matplotlib for compatibility over Open3D's visualizer.
"""

import numpy as np
import open3d as o3d
import os
from PIL import Image
import matplotlib.pyplot as plt
from pipeline.config import DEFAULT_NUM_SAMPLES, ENABLE_VISUALIZATION

class MeshVisualizer:
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai

    def show(self, title="Visualization", save_path=None):
        """Render comparison view with robust sampling."""
        if not ENABLE_VISUALIZATION and save_path is None:
            return

        try:
            # Try to sample points, fallback to vertices if sampling fails
            try:
                points_gt = np.array(self.mesh_gt.sample(DEFAULT_NUM_SAMPLES))
            except:
                points_gt = np.array(self.mesh_gt.vertices)
                points_gt = points_gt[np.random.choice(len(points_gt), 
                                    min(DEFAULT_NUM_SAMPLES, len(points_gt)))]

            try:
                points_ai = np.array(self.mesh_ai.sample(DEFAULT_NUM_SAMPLES))
            except:
                points_ai = np.array(self.mesh_ai.vertices)
                points_ai = points_ai[np.random.choice(len(points_ai), 
                                    min(DEFAULT_NUM_SAMPLES, len(points_ai)))]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points_gt[:, 0], points_gt[:, 1], points_gt[:, 2], 
                    c='red', s=1, label='GT')
            ax.scatter(points_ai[:, 0], points_ai[:, 1], points_ai[:, 2], 
                    c='green', s=1, label='AI')
            ax.set_title(title)
            ax.legend()
            ax.view_init(elev=15, azim=45)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            elif ENABLE_VISUALIZATION:
                plt.show()

        except Exception as e:
            print(f"Visualization failed: {e}")
