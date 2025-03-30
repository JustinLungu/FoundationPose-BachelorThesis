import open3d as o3d
import numpy as np

class PointCloudInspector:
    def __init__(self, ply_path):
        self.ply_path = ply_path
        self.pcd = self._load_point_cloud(self.ply_path)
    
    def _load_point_cloud(self, file_path):
        return o3d.io.read_point_cloud(file_path)
    
    def get_dimensions(self):
        bbox = self.pcd.get_axis_aligned_bounding_box()
        size = bbox.get_extent()  # [width, height, depth]
        max_dim = np.max(size)
        return size, max_dim
    
    def print_dimensions(self):
        size, max_dim = self.get_dimensions()
        print(f"Object size (W, H, D) in meters: {size}")
        print(f"Max dimension: {max_dim} meters")

if __name__ == "__main__":
    inspector = PointCloudInspector("obj_01.ply")
    inspector.print_dimensions()
