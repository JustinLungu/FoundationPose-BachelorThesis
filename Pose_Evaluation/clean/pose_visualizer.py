import open3d as o3d
import numpy as np
import yaml

class PoseVisualizer:
    def __init__(self, gt_yaml_path, res_yaml_path, ply_path):
        self.gt_yaml_path = gt_yaml_path
        self.res_yaml_path = res_yaml_path
        self.ply_path = ply_path
        self.gt_matrices = self._load_yaml(self.gt_yaml_path)
        self.res_matrices = self._load_yaml(self.res_yaml_path)
        self.pcd = self._load_point_cloud(self.ply_path)
    
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

    def _load_point_cloud(self, file_path):
        return o3d.io.read_point_cloud(file_path)
    
    def visualize_frame(self, frame_index=0):
        if len(self.gt_matrices) == 0 or len(self.res_matrices) == 0:
            raise ValueError("No transformation matrices found!")
        
        T_gt = self.gt_matrices[frame_index]
        T_pred = self.res_matrices[frame_index]
        points = np.asarray(self.pcd.points)

        # Apply transformation
        transformed_gt = (T_gt[:3, :3] @ points.T).T + T_gt[:3, 3]
        transformed_pred = (T_pred[:3, :3] @ points.T).T + T_pred[:3, 3]

        # Create colored point clouds
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(transformed_gt)
        pcd_gt.paint_uniform_color([1, 0, 0])  # red

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(transformed_pred)
        pcd_pred.paint_uniform_color([0, 1, 0])  # green

        # Visualize both point clouds together
        o3d.visualization.draw_geometries([pcd_gt, pcd_pred])

if __name__ == "__main__":
    visualizer = PoseVisualizer("gt_reformatted.yml", "res_reformatted.yml", "obj_01.ply")
    visualizer.visualize_frame()
