import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml

from PIL import Image, ImageDraw, ImageFont
import os
import imageio

from open3d.visualization import rendering

class TransformationVisualizer:
    def __init__(self, rotation_errors, translation_errors, pose_errors, add_errors):
        self.rotation_errors = rotation_errors
        self.translation_errors = translation_errors
        self.pose_errors = pose_errors
        self.add_errors = add_errors
        self.frames = np.arange(len(rotation_errors))

    def plot_outliers(self):
        rot_th, trans_th, pose_th, add_th = 10, 0.05, 0.1, 0.05
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        self._scatter_plot(axs[0], self.rotation_errors, rot_th, "Rotation Error Outliers", "Degrees", "blue")
        self._scatter_plot(axs[1], self.translation_errors, trans_th, "Translation Error Outliers", "Meters", "orange")
        self._scatter_plot(axs[2], self.pose_errors, pose_th, "Pose Error Outliers", "Error", "green")
        self._scatter_plot(axs[3], self.add_errors, add_th, "ADD Error Outliers", "Meters", "purple")

        plt.tight_layout()
        plt.savefig("plots/error_outliers.png")
        plt.close()

    def plot_trends(self):
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        self._trend_plot(axs[0], self.rotation_errors, [5, 10], "Rotation Error Over Frames", "Degrees", "blue")
        self._trend_plot(axs[1], self.translation_errors, [0.01, 0.05], "Translation Error Over Frames", "Meters", "orange")
        self._trend_plot(axs[2], self.pose_errors, [0.1, 0.3], "Pose Error Over Frames", "Error", "green")
        self._trend_plot(axs[3], self.add_errors, [0.01, 0.05], "ADD Error Over Frames", "Meters", "purple")

        plt.tight_layout()
        plt.savefig("plots/error_trends.png")
        plt.close()

    def plot_distributions(self):
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        self._histogram(axs[0], self.rotation_errors, [5, 10], "Rotation Error Distribution", "Degrees", "blue")
        self._histogram(axs[1], self.translation_errors, [0.01, 0.05], "Translation Error Distribution", "Meters", "orange")
        self._histogram(axs[2], self.pose_errors, [0.1, 0.3], "Pose Error Distribution", "Error", "green")
        self._histogram(axs[3], self.add_errors, [0.01, 0.05], "ADD Error Distribution", "Meters", "purple")

        plt.tight_layout()
        plt.savefig("plots/error_distributions.png")
        plt.close()

    def _scatter_plot(self, ax, errors, threshold, title, ylabel, color):
        ax.scatter(self.frames, errors, color=color, alpha=0.5, label=title.replace(" Outliers", ""))
        ax.scatter([i for i in self.frames if errors[i] > threshold],
                   [errors[i] for i in self.frames if errors[i] > threshold],
                   color="red", label=f"Outlier (>{threshold})")
        ax.set_title(title)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel(ylabel)
        ax.legend()

    def _trend_plot(self, ax, errors, thresholds, title, ylabel, color):
        ax.plot(self.frames, errors, label=title.replace(" Over Frames", ""), color=color)
        ax.axhline(thresholds[0], color='green', linestyle='--', label=f'Good (<{thresholds[0]})')
        ax.axhline(thresholds[1], color='red', linestyle='--', label=f'Bad (>{thresholds[1]})')
        ax.set_title(title)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel(ylabel)
        ax.legend()

    def _histogram(self, ax, errors, thresholds, title, ylabel, color):
        ax.hist(errors, bins=50, color=color, alpha=0.7)
        ax.axvline(thresholds[0], color='green', linestyle='--', label=f'Good (<{thresholds[0]})')
        ax.axvline(thresholds[1], color='red', linestyle='--', label=f'Bad (>{thresholds[1]})')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()


import open3d as o3d
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
import imageio
from open3d.visualization import rendering

class AlignmentVisualizer:
    def __init__(self, gt_path, pred_path, ply_path):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.ply_path = ply_path

        self.gt_matrices = self._load_yaml(self.gt_path)
        self.res_matrices = self._load_yaml(self.pred_path)

        if len(self.gt_matrices) == 0 or len(self.res_matrices) == 0:
            raise ValueError("No transformation matrices found!")

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

    def _get_transformed_pcds(self, frame_index):
        pcd = o3d.io.read_point_cloud(self.ply_path)
        points = np.asarray(pcd.points)

        T_gt = self.gt_matrices[frame_index]
        T_pred = self.res_matrices[frame_index]

        transformed_gt = (T_gt[:3, :3] @ points.T).T + T_gt[:3, 3]
        transformed_pred = (T_pred[:3, :3] @ points.T).T + T_pred[:3, 3]

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(transformed_gt)
        pcd_gt.paint_uniform_color([1, 0, 0])

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(transformed_pred)
        pcd_pred.paint_uniform_color([0, 1, 0])

        return pcd_gt, pcd_pred, transformed_gt, transformed_pred

    def save_alignment_image(self, output_path, frame_index=0, width=1920, height=1080, zoomed=False, zoom_factor=0.3):
        pcd_gt, pcd_pred, transformed_gt, transformed_pred = self._get_transformed_pcds(frame_index)

        renderer = rendering.OffscreenRenderer(width, height)
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"

        scene.add_geometry("gt", pcd_gt, mat)
        scene.add_geometry("pred", pcd_pred, mat)

        all_points = np.vstack((transformed_gt, transformed_pred))
        center = np.mean(all_points, axis=0)
        radius = np.linalg.norm(np.max(all_points, axis=0) - np.min(all_points, axis=0))
        distance = radius * zoom_factor if zoomed else radius * 2.5
        eye = center + np.array([distance, distance, distance])
        scene.camera.look_at(center, eye, [0, 0, 1])

        img = renderer.render_to_image()
        o3d.io.write_image(output_path, img)
        print(f"[✓] Saved {'zoomed' if zoomed else 'full'} view to {output_path}")

    def save_annotated_image(self, base_img_path, output_path, frame_index, errors):
        img = Image.open(base_img_path)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        except:
            font = ImageFont.load_default()

        text = f"Frame: {frame_index}\n"
        text += f"Rotation Error: {errors['Rotation Error (deg)'][frame_index]:.2f}°\n"
        text += f"Translation Error: {errors['Translation Error (m)'][frame_index]:.4f}m\n"
        text += f"ADD: {errors['ADD (m)'][frame_index]:.4f}m"

        draw.text((50, 50), text, font=font, fill=(0, 0, 0))
        img.save(output_path)
        print(f"[✓] Saved annotated image to {output_path}")

    def save_orbit_gif(self, frame_index=0, output_path="orbit.gif", n_frames=36, zoom_factor=0.4):
        pcd_gt, pcd_pred, transformed_gt, transformed_pred = self._get_transformed_pcds(frame_index)

        renderer = rendering.OffscreenRenderer(640, 480)
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        scene.add_geometry("gt", pcd_gt, mat)
        scene.add_geometry("pred", pcd_pred, mat)

        all_points = np.vstack((transformed_gt, transformed_pred))
        center = np.mean(all_points, axis=0)
        radius = np.linalg.norm(np.max(all_points, axis=0) - np.min(all_points, axis=0)) * zoom_factor

        images = []
        for i in range(n_frames):
            theta = (2 * np.pi * i) / n_frames
            eye = center + radius * np.array([np.cos(theta), np.sin(theta), 0.5])
            scene.camera.look_at(center, eye, [0, 0, 1])
            img = renderer.render_to_image()
            np_img = np.asarray(img)
            images.append(np_img)

        imageio.mimsave(output_path, images, fps=12)
        print(f"[\u2713] Saved orbiting camera GIF to {output_path}")
