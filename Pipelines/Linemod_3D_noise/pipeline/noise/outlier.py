"""
Outlier Noise Implementation

Perturbs only a random subset of vertices with strong noise. This simulates 
scanning artifacts like dust particles, small occlusions, or data corruption.

Mathematical Formulation:
    Select k = percentage * num_vertices
    For each selected vertex:
        vertex += N(0, σ_outlier)  # σ_outlier >> typical noise σ

Characteristics:
    - Creates localized spikes rather than uniform noise
    - Tests robustness to catastrophic corruption
    - Percentage controls sparsity (typical 1-5%)
    - Large σ_outlier creates noticeable defects
"""

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