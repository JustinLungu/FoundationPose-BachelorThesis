import numpy as np
import yaml
import matplotlib.pyplot as plt
import open3d as o3d


class TransformationEvaluator:
    def __init__(self, gt_file, res_file, ply_file):
        self.gt_matrices = self.load_yaml(gt_file)
        self.res_matrices = self.load_yaml(res_file)
        self.object_points = self.load_ply(ply_file)
        self.num_matrices = len(self.gt_matrices)

    def load_yaml(self, file_path):
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

    def load_ply(self, ply_file):
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)
        return points / 1000.0  # Convert mm to meters

    def compute_rotation_error(self, R_gt, R_pred):
        error = np.arccos((np.trace(R_gt.T @ R_pred) - 1) / 2)
        return np.degrees(error)

    def compute_translation_error(self, t_gt, t_pred):
        return np.linalg.norm(t_gt - t_pred)

    def compute_pose_error(self, T_gt, T_pred):
        return np.linalg.norm(T_gt - T_pred, ord='fro')

    def compute_add(self, T_gt, T_pred):
        transformed_gt = (T_gt[:3, :3] @ self.object_points.T).T + T_gt[:3, 3]
        transformed_pred = (T_pred[:3, :3] @ self.object_points.T).T + T_pred[:3, 3]
        return np.mean(np.linalg.norm(transformed_gt - transformed_pred, axis=1))

    def evaluate(self):
        rotation_errors, translation_errors, pose_errors, add_errors = [], [], [], []

        for i in range(self.num_matrices):
            T_gt = self.gt_matrices[i]
            T_pred = self.res_matrices[i]

            R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
            R_pred, t_pred = T_pred[:3, :3], T_pred[:3, 3]

            rotation_errors.append(self.compute_rotation_error(R_gt, R_pred))
            translation_errors.append(self.compute_translation_error(t_gt, t_pred))
            pose_errors.append(self.compute_pose_error(T_gt, T_pred))
            add_errors.append(self.compute_add(T_gt, T_pred))

        return {
            "Rotation Error (deg)": rotation_errors,
            "Translation Error (m)": translation_errors,
            "Pose Error (Frobenius norm)": pose_errors,
            "ADD (m)": add_errors
        }
