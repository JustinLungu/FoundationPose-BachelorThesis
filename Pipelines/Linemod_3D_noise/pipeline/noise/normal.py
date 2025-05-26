"""
Normal-Directional Noise Implementation

Perturbs vertices along their surface normal directions only. 
This simulates manufacturing imperfections or surface erosion where defects 
tend to follow the surface contour rather than random spatial directions.

Mathematical Formulation:
    displacement = surface_normal * N(0, σ)
    noisy_vertex = vertex + displacement

Characteristics:
    - Noise follows mesh curvature
    - Preserves sharp edges better than isotropic noise
    - Simulates directional wear/abrasion
    - Particularly relevant for 3D-scanned objects
"""

import numpy as np
import open3d as o3d
from .base_noise import BaseNoise

class NormalNoise(BaseNoise):
    def __init__(self, std_dev=0.001):
        self.std_dev = std_dev

    def apply(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        displacement = normals * np.random.normal(0, self.std_dev, normals.shape)
        mesh.vertices = o3d.utility.Vector3dVector(vertices + displacement)
        return mesh