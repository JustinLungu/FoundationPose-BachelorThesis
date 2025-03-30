import numpy as np
import yaml
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R

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

        # Convert from mm to meters
        points /= 1000.0

        return points


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
        rotation_errors = []
        translation_errors = []
        pose_errors = []
        add_errors = []

        for i in range(self.num_matrices):
            T_gt = self.gt_matrices[i]
            T_pred = self.res_matrices[i]

            R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
            R_pred, t_pred = T_pred[:3, :3], T_pred[:3, 3]

            rotation_errors.append(self.compute_rotation_error(R_gt, R_pred))
            translation_errors.append(self.compute_translation_error(t_gt, t_pred))
            pose_errors.append(self.compute_pose_error(T_gt, T_pred))
            add_errors.append(self.compute_add(T_gt, T_pred))

        self.plot_results(rotation_errors, translation_errors, pose_errors, add_errors)

        return {
            "Rotation Error (deg)": np.mean(rotation_errors),
            "Translation Error (m)": np.mean(translation_errors),
            "Pose Error (Frobenius norm)": np.mean(pose_errors),
            "ADD (m)": np.mean(add_errors)
        }

    def plot_results(self, rotation_errors, translation_errors, pose_errors, add_errors):
        frames = np.arange(len(rotation_errors))
        
        # Outlier threshold
        rot_threshold = 10
        trans_threshold = 0.05
        pose_threshold = 0.1
        add_threshold = 0.05
        
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))
        
        axs[0].scatter(frames, rotation_errors, color="blue", alpha=0.5, label="Rotation Error")
        axs[0].scatter([i for i in frames if rotation_errors[i] > rot_threshold],
                    [rotation_errors[i] for i in frames if rotation_errors[i] > rot_threshold],
                    color="red", label="Outlier (>10°)")
        axs[0].set_title("Rotation Error Outliers")
        axs[0].set_xlabel("Frame Index")
        axs[0].set_ylabel("Degrees")
        axs[0].legend()
        
        axs[1].scatter(frames, translation_errors, color="orange", alpha=0.5, label="Translation Error")
        axs[1].scatter([i for i in frames if translation_errors[i] > trans_threshold],
                    [translation_errors[i] for i in frames if translation_errors[i] > trans_threshold],
                    color="red", label="Outlier (>0.05m)")
        axs[1].set_title("Translation Error Outliers")
        axs[1].set_xlabel("Frame Index")
        axs[1].set_ylabel("Meters")
        axs[1].legend()
        
        axs[2].scatter(frames, pose_errors, color="green", alpha=0.5, label="Pose Error")
        axs[2].scatter([i for i in frames if pose_errors[i] > pose_threshold],
                    [pose_errors[i] for i in frames if pose_errors[i] > pose_threshold],
                    color="red", label="Outlier (>0.1)")
        axs[2].set_title("Pose Error Outliers")
        axs[2].set_xlabel("Frame Index")
        axs[2].set_ylabel("Error")
        axs[2].legend()
        
        axs[3].scatter(frames, add_errors, color="purple", alpha=0.5, label="ADD Error")
        axs[3].scatter([i for i in frames if add_errors[i] > add_threshold],
                    [add_errors[i] for i in frames if add_errors[i] > add_threshold],
                    color="red", label="Outlier (>0.05m)")
        axs[3].set_title("ADD Error Outliers")
        axs[3].set_xlabel("Frame Index")
        axs[3].set_ylabel("Meters")
        axs[3].legend()
        
        plt.tight_layout()
        plt.savefig("error_outliers.png")
        plt.close()
        
        # Trend Plot
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        axs[0].plot(frames, rotation_errors, label="Rotation Error (deg)", color="blue")
        axs[0].axhline(5, color='green', linestyle='--', label='Good (<5°)')
        axs[0].axhline(10, color='red', linestyle='--', label='Bad (>10°)')
        axs[0].set_title("Rotation Error Over Frames")
        axs[0].set_xlabel("Frame Index")
        axs[0].set_ylabel("Degrees")
        axs[0].legend()

        axs[1].plot(frames, translation_errors, label="Translation Error (m)", color="orange")
        axs[1].axhline(0.01, color='green', linestyle='--', label='Good (<0.01m)')
        axs[1].axhline(0.05, color='red', linestyle='--', label='Bad (>0.05m)')
        axs[1].set_title("Translation Error Over Frames")
        axs[1].set_xlabel("Frame Index")
        axs[1].set_ylabel("Meters")
        axs[1].legend()

        axs[2].plot(frames, pose_errors, label="Pose Error (Frobenius norm)", color="green")
        axs[2].axhline(0.1, color='green', linestyle='--', label='Good (<0.1)')
        axs[2].axhline(0.3, color='red', linestyle='--', label='Bad (>0.3)')
        axs[2].set_title("Pose Error Over Frames")
        axs[2].set_xlabel("Frame Index")
        axs[2].set_ylabel("Error")
        axs[2].legend()
        
        axs[3].plot(frames, add_errors, label="ADD Error (m)", color="purple")
        axs[3].axhline(0.01, color='green', linestyle='--', label='Good (<0.01m)')
        axs[3].axhline(0.05, color='red', linestyle='--', label='Bad (>0.05m)')
        axs[3].set_title("ADD Error Over Frames")
        axs[3].set_xlabel("Frame Index")
        axs[3].set_ylabel("Meters")
        axs[3].legend()

        plt.tight_layout()
        plt.savefig("error_trends.png")
        plt.close()
        
        # Histogram Distribution
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        axs[0].hist(rotation_errors, bins=50, color="blue", alpha=0.7)
        axs[0].axvline(5, color='green', linestyle='--', label='Good (<5°)')
        axs[0].axvline(10, color='red', linestyle='--', label='Bad (>10°)')
        axs[0].set_title("Rotation Error Distribution")
        axs[0].legend()

        axs[1].hist(translation_errors, bins=50, color="orange", alpha=0.7)
        axs[1].axvline(0.01, color='green', linestyle='--', label='Good (<0.01m)')
        axs[1].axvline(0.05, color='red', linestyle='--', label='Bad (>0.05m)')
        axs[1].set_title("Translation Error Distribution")
        axs[1].legend()

        axs[2].hist(pose_errors, bins=50, color="green", alpha=0.7)
        axs[2].axvline(0.1, color='green', linestyle='--', label='Good (<0.1)')
        axs[2].axvline(0.3, color='red', linestyle='--', label='Bad (>0.3)')
        axs[2].set_title("Pose Error Distribution")
        axs[2].legend()

        axs[3].hist(add_errors, bins=50, color="purple", alpha=0.7)
        axs[3].axvline(0.01, color='green', linestyle='--', label='Good (<0.01m)')
        axs[3].axvline(0.05, color='red', linestyle='--', label='Bad (>0.05m)')
        axs[3].set_title("ADD Error Distribution")
        axs[3].legend()

        plt.tight_layout()
        plt.savefig("error_distributions.png")
        plt.close()


if __name__ == "__main__":
    evaluator = TransformationEvaluator("gt_reformatted.yml", "res_reformatted.yml", "obj_01.ply")
    results = evaluator.evaluate()
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
