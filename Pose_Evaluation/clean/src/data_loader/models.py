from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class PointCloud:
    """3D point cloud data container"""
    points: np.ndarray  # Shape (N, 3)
    units: str = "meters"
    
    def __post_init__(self):
        if len(self.points.shape) != 2 or self.points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")

@dataclass
class Transformation:
    """4x4 transformation matrix container"""
    matrix: np.ndarray
    frame_id: int
    obj_id: Optional[str] = None
    
    def __post_init__(self):
        if self.matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")
        if not isinstance(self.frame_id, int):
            raise TypeError("Frame ID must be integer")