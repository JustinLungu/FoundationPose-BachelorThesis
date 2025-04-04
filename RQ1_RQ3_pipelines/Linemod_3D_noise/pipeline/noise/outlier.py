import numpy as np
import open3d as o3d
from .base_noise import BaseNoise

class OutlierNoise(BaseNoise):
    def __init__(self, percentage=0.02, std_dev=0.01):
        self.percentage = percentage
        self.std_dev = std_dev

    def apply(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        vertices = np.asarray(mesh.vertices).copy()
        num_vertices = len(vertices)
        num_outliers = int(self.percentage * num_vertices)
        indices = np.random.choice(num_vertices, num_outliers, replace=False)
        noise = np.random.normal(0, self.std_dev, (num_outliers, 3))
        vertices[indices] += noise
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        return mesh