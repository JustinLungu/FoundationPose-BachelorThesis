import open3d as o3d
import numpy as np
import yaml


class SimpleAlignmentVisualizer:
    """
    This visualization shows the alignment between the ground truth (red) and predicted (green)
    point clouds after applying the transformation matrices.
    
    Interpretation:
    - If red and green points perfectly overlap, the estimated transformation is very accurate.
    - Small offset = low error but not perfect.
    """
    def __init__(self, gt_path, pred_path, ply_path):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.ply_path = ply_path

        self.gt_matrices = self._load_yaml(self.gt_path)
        self.res_matrices = self._load_yaml(self.pred_path)

        if len(self.gt_matrices) == 0 or len(self.res_matrices) == 0:
            raise ValueError("No transformation matrices found!")

        self.T_gt = self.gt_matrices[0]
        self.T_pred = self.res_matrices[0]

    def _load_yaml(self, file_path):
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

    def visualize(self):
        # Load PLY file
        pcd = o3d.io.read_point_cloud(self.ply_path)
        points = np.asarray(pcd.points)

        # Transform points
        transformed_gt = (self.T_gt[:3, :3] @ points.T).T + self.T_gt[:3, 3]
        transformed_pred = (self.T_pred[:3, :3] @ points.T).T + self.T_pred[:3, 3]

        # Create Open3D point clouds
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(transformed_gt)
        pcd_gt.paint_uniform_color([1, 0, 0])  # Red for ground truth

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(transformed_pred)
        pcd_pred.paint_uniform_color([0, 1, 0])  # Green for predicted

        # Visualize the misalignment
        o3d.visualization.draw_geometries([pcd_gt, pcd_pred])


if __name__ == "__main__":
    visualizer = SimpleAlignmentVisualizer(
        "gt_reformatted.yml",
        "res_reformatted.yml",
        "obj_01.ply"
    )
    visualizer.visualize()
