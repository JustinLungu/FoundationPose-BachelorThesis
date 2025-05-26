"""
Speckle (Multiplicative) Noise Implementation

Applies noise proportional to vertex distance from origin. This creates 
relative rather than absolute distortion - features farther from center 
get larger absolute displacements but similar relative distortion.

Mathematical Formulation:
    noise = vertex * N(0, σ)  
    noisy_vertex = vertex + noise

Characteristics:
    - Preserves relative shape proportions
    - Simulates lens distortion effects
    - More distortion in larger mesh regions
    - Useful for testing scale-invariant algorithms
"""

import numpy as np
import open3d as o3d
from .base_noise import BaseNoise

class SpeckleNoise(BaseNoise):
    def __init__(self, std_dev=0.001):
        self.std_dev = std_dev

    def apply(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        vertices = np.asarray(mesh.vertices)
        noise = vertices * np.random.normal(0, self.std_dev, vertices.shape)
        mesh.vertices = o3d.utility.Vector3dVector(vertices + noise)
        return mesh