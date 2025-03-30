import yaml
import open3d as o3d
import numpy as np
from typing import List
from .models import PointCloud, Transformation

class DataLoader:
    @staticmethod
    def load_yaml(path: str) -> List[Transformation]:
        """Load transformations from YAML file"""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        transforms = []
        for obj_id, frames in data.items():
            for frame_id, matrices in frames.items():
                for name, matrix in matrices.items():
                    try:
                        arr = np.array(matrix)
                        if arr.shape == (4, 4):
                            transforms.append(
                                Transformation(arr, int(frame_id), str(obj_id))
                            )
                    except Exception as e:
                        print(f"Skipping invalid matrix {obj_id}-{frame_id}-{name}: {e}")
        return transforms

    @staticmethod
    def load_ply(path: str, scale: float = 0.001) -> PointCloud:
        """Load and scale PLY file to meters"""
        pcd = o3d.io.read_point_cloud(path)
        return PointCloud(np.asarray(pcd.points) * scale)