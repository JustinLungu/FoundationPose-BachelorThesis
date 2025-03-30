import open3d as o3d
import numpy as np
import yaml

"""
This visualization shows the alignment between the ground truth (red) and predicted (green) point clouds after applying the transformation matrices.

Interpretation:

If the red and green points perfectly overlap, the estimated transformation is very accurate.
If there’s a small offset, the error is low, but not perfect.
"""

# Load transformation matrices from YAML
def load_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    matrices = []
    for outer_key in data:
        for inner_key in data[outer_key]:
            for matrix_key in data[outer_key][inner_key]:
                try:
                    matrix = np.array(data[outer_key][inner_key][matrix_key])
                    if matrix.shape == (4, 4):
                        matrices.append(matrix)
                except Exception as e:
                    print(f"Error loading matrix {outer_key}-{inner_key}-{matrix_key}: {e}")
    return matrices

# Load transformation matrices
gt_matrices = load_yaml("gt_reformatted.yml")
res_matrices = load_yaml("res_reformatted.yml")

# Ensure we have at least one transformation to visualize
if len(gt_matrices) == 0 or len(res_matrices) == 0:
    raise ValueError("No transformation matrices found!")

# Use the first frame as an example
T_gt = gt_matrices[0]
T_pred = res_matrices[0]

# Load PLY file
pcd = o3d.io.read_point_cloud("obj_01.ply")
points = np.asarray(pcd.points)

# Transform points
transformed_gt = (T_gt[:3, :3] @ points.T).T + T_gt[:3, 3]
transformed_pred = (T_pred[:3, :3] @ points.T).T + T_pred[:3, 3]

# Create Open3D point clouds
pcd_gt = o3d.geometry.PointCloud()
pcd_gt.points = o3d.utility.Vector3dVector(transformed_gt)
pcd_gt.paint_uniform_color([1, 0, 0])  # Red for ground truth

pcd_pred = o3d.geometry.PointCloud()
pcd_pred.points = o3d.utility.Vector3dVector(transformed_pred)
pcd_pred.paint_uniform_color([0, 1, 0])  # Green for predicted

# Visualize the misalignment
o3d.visualization.draw_geometries([pcd_gt, pcd_pred])
