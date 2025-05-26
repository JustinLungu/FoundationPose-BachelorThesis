"""
Gaussian (Isotropic) Noise Implementation

Applies independent Gaussian noise to each vertex coordinate equally in all directions.
This simulates general sensor noise or measurement inaccuracies that affect the mesh uniformly.

Mathematical Formulation:
    noisy_vertex = vertex + N(μ, σ) 
    where N = 3D normal distribution (x,y,z components sampled independently)

Characteristics:
    - Equal perturbation in all spatial directions
    - Preserves overall shape but adds "fuzziness"
    - Magnitude controlled by standard deviation (σ)
    - Good baseline for general noise robustness testing
"""

import numpy as np
import open3d as o3d
from .base_noise import BaseNoise

class GaussianNoise(BaseNoise):
    def __init__(self, mean=0.0, std_dev=0.001):
        self.mean = mean
        self.std_dev = std_dev

    def apply(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        vertices = np.asarray(mesh.vertices)
        noise = np.random.normal(self.mean, self.std_dev, vertices.shape)
        noisy_vertices = vertices + noise
        mesh.vertices = o3d.utility.Vector3dVector(noisy_vertices)
        return mesh
