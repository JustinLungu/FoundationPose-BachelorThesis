import open3d as o3d
import numpy as np

# Load your PLY file
pcd = o3d.io.read_point_cloud("obj_01.ply")

# Get the bounding box
bbox = pcd.get_axis_aligned_bounding_box()
size = bbox.get_extent()  # Returns [width, height, depth]

print(f"Object size (W, H, D) in meters: {size}")
print(f"Max dimension: {np.max(size)} meters")
