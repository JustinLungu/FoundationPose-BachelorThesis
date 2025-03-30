import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml
import imageio
from pipeline import config as cfg
from PIL import Image, ImageDraw, ImageFont


from open3d.visualization import rendering

class TransformationVisualizer:
    def __init__(self, rotation_errors, translation_errors, pose_errors, add_errors):
        self.rotation_errors = rotation_errors
        self.translation_errors = translation_errors
        self.pose_errors = pose_errors
        self.add_errors = add_errors
        self.frames = np.arange(len(rotation_errors))

    def plot_outliers(self, output_path):
        thresholds = cfg.OUTLIER_THRESHOLDS
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        for i, (title, ylabel, color, _) in enumerate(cfg.LABELS):
            self._scatter_plot(axs[i], getattr(self, f"{title.lower().split()[0]}_errors"),
                               thresholds[i], f"{title} Outliers", ylabel, color)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_trends(self, output_path):
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        for i, (title, ylabel, color, key) in enumerate(cfg.LABELS):
            self._trend_plot(axs[i], getattr(self, f"{key}_errors"),
                             cfg.TREND_THRESHOLDS[key],
                             f"{title} Over Frames", ylabel, color)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_distributions(self, output_path):
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        for i, (title, ylabel, color, key) in enumerate(cfg.LABELS):
            self._histogram(axs[i], getattr(self, f"{key}_errors"),
                            cfg.TREND_THRESHOLDS[key],
                            f"{title} Distribution", ylabel, color)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _scatter_plot(self, ax, errors, threshold, title, ylabel, color):
        ax.scatter(self.frames, errors, color=color, alpha=0.5)
        ax.scatter([i for i in self.frames if errors[i] > threshold],
                   [errors[i] for i in self.frames if errors[i] > threshold],
                   color="red", label=f"Outlier (>{threshold})")
        ax.set_title(title)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel(ylabel)
        ax.legend()

    def _trend_plot(self, ax, errors, thresholds, title, ylabel, color):
        ax.plot(self.frames, errors, color=color)
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

class AlignmentVisualizer:
    def __init__(self, gt_path, pred_path, ply_path, rotation_angles=None):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.ply_path = ply_path
        self.rotation_angles = rotation_angles  # (rx, ry, rz) in degrees

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

        # Apply rotation from main (in degrees)
        if self.rotation_angles is not None:
            rx, ry, rz = [np.radians(a) for a in self.rotation_angles]
            R = pcd.get_rotation_matrix_from_xyz((rx, ry, rz))
            pcd.rotate(R, center=(0, 0, 0))

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
    

    def show_interactive(self, frame_index=0):
        pcd_gt, pcd_pred, *_ = self._get_transformed_pcds(frame_index)
        o3d.visualization.draw_geometries([pcd_gt, pcd_pred])

    def save_alignment_image(self, output_path, frame_index=0, width=1920, height=1080, zoom_factor=0.4):
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
        distance = radius * zoom_factor
        eye = center + np.array([distance, distance, distance])
        scene.camera.look_at(center, eye, [0, 0, 1])

        img = renderer.render_to_image()
        o3d.io.write_image(output_path, img)
        print(f"[✓] Saved image to {output_path}")

    def save_annotated_image(self, base_img_path, output_path, frame_index, errors):
        img = Image.open(base_img_path)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        except:
            font = ImageFont.load_default()

        text_block = f"Frame: {frame_index}\n"
        text_block += f"Rotation Error: {errors['Rotation Error (deg)'][frame_index]:.2f}°\n"
        text_block += f"Translation Error: {errors['Translation Error (m)'][frame_index]:.4f}m\n"
        text_block += f"ADD: {errors['ADD (m)'][frame_index]:.4f}m"

        img = self._add_legend(img, text_block=text_block)
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
            img_with_legend = self._add_legend(Image.fromarray(np_img))
            images.append(np.asarray(img_with_legend))

        imageio.mimsave(output_path, images, fps=12)
        print(f"[✓] Saved orbiting camera GIF to {output_path}")

    def _add_legend(self, img, text_block=None):
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        except:
            font = ImageFont.load_default()

        legend_items = [
            ("Ground Truth", (255, 0, 0)),
            ("Prediction", (0, 255, 0))
        ]

        swatch_size = 40
        spacing = 8
        padding = 10

        img_width, img_height = img.size
        y = img_height - padding - (swatch_size + spacing) * len(legend_items)

        for label, color in legend_items:
            draw.rectangle([padding, y, padding + swatch_size, y + swatch_size], fill=color)
            draw.text((padding + swatch_size + 10, y), label, fill=(0, 0, 0), font=font)
            y += swatch_size + spacing

        if text_block:
            lines = text_block.strip().split("\n")
            y -= (swatch_size + spacing) * len(legend_items) + len(lines)*(font.size + 5) + 10
            for line in lines:
                draw.text((padding, y), line, font=font, fill=(0, 0, 0))
                y += font.size + 5

        return img
    
    def show_interactive(self, frame_index=0, azimuth=0, elevation=0, distance_factor=2.5):
        import open3d as o3d
        from math import cos, sin, radians
        import numpy as np

        # Get transformed point clouds
        _, _, transformed_gt, transformed_pred = self._get_transformed_pcds(frame_index)

        # Combine all points to find center and radius
        all_points = np.vstack((transformed_gt, transformed_pred))
        center = np.mean(all_points, axis=0)
        radius = np.linalg.norm(np.max(all_points, axis=0) - np.min(all_points, axis=0))
        distance = radius * distance_factor

        # Create point cloud geometries
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(transformed_gt)
        pcd_gt.paint_uniform_color([1, 0, 0])  # red

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(transformed_pred)
        pcd_pred.paint_uniform_color([0, 1, 0])  # green

        # Calculate eye position in spherical coords
        az = radians(azimuth)
        el = radians(elevation)
        eye = center + distance * np.array([
            cos(el) * cos(az),
            cos(el) * sin(az),
            sin(el)
        ])

        front = (center - eye)
        front /= np.linalg.norm(front)

        # Show in visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_gt)
        vis.add_geometry(pcd_pred)

        ctr = vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front(front)
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.7)

        vis.run()
        vis.destroy_window()
