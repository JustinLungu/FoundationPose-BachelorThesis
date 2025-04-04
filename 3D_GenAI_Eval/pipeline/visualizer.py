import numpy as np
import open3d as o3d
import os
from PIL import Image
import matplotlib.pyplot as plt
from constants import DEFAULT_NUM_SAMPLES, ENABLE_VISUALIZATION

class MeshVisualizer:
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai

    def show(self, title="Visualization", save_path=None):
        if not ENABLE_VISUALIZATION and save_path is None:
            return

        points_gt = np.array(self.mesh_gt.sample(DEFAULT_NUM_SAMPLES))
        points_ai = np.array(self.mesh_ai.sample(DEFAULT_NUM_SAMPLES))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_gt[:, 0], points_gt[:, 1], points_gt[:, 2], c='red', s=1, label='GT')
        ax.scatter(points_ai[:, 0], points_ai[:, 1], points_ai[:, 2], c='green', s=1, label='AI')
        ax.set_title(title)
        ax.legend()
        ax.view_init(elev=15, azim=45)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        elif ENABLE_VISUALIZATION:
            plt.show()
