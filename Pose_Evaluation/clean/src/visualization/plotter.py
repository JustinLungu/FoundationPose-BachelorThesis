import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import open3d as o3d

class Plotter:
    def __init__(self):
        self.figure_size = (12, 12)
        self.dpi = 300

    def save_3d_visualization(self, pcd_gt, pcd_pred, save_path):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        vis.add_geometry(pcd_gt)
        vis.add_geometry(pcd_pred)
        vis.capture_screen_image(str(save_path))
        vis.destroy_window()

    def create_point_clouds(self, T_gt, T_pred, points):
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(
            (T_gt[:3, :3] @ points.T).T + T_gt[:3, 3])
        pcd_gt.paint_uniform_color([1, 0, 0])  # Red
        
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(
            (T_pred[:3, :3] @ points.T).T + T_pred[:3, 3])
        pcd_pred.paint_uniform_color([0, 1, 0])  # Green
        
        return pcd_gt, pcd_pred

    def generate_all_plots(self, evaluation_results, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        errors = evaluation_results["errors"]
        self._plot_outliers(errors, save_dir / "outliers.png")
        self._plot_trends(errors, save_dir / "trends.png")
        self._plot_histograms(errors, save_dir / "histograms.png")

    def _plot_outliers(self, errors, save_path):
        fig, axs = plt.subplots(4, 1, figsize=self.figure_size)
        frames = np.arange(len(errors["rotation"]))
        
        # Rotation outliers
        axs[0].scatter(frames, errors["rotation"], alpha=0.5)
        axs[0].set_title("Rotation Error Outliers")
        # ... (similar for other metrics)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

    def _plot_trends(self, errors, save_path):
        fig, axs = plt.subplots(4, 1, figsize=self.figure_size)
        frames = np.arange(len(errors["rotation"]))
        
        # Rotation trend
        axs[0].plot(frames, errors["rotation"])
        axs[0].set_title("Rotation Error Trend")
        # ... (similar for other metrics)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

    def _plot_histograms(self, errors, save_path):
        fig, axs = plt.subplots(4, 1, figsize=self.figure_size)
        
        # Rotation histogram
        axs[0].hist(errors["rotation"], bins=50)
        axs[0].set_title("Rotation Error Distribution")
        # ... (similar for other metrics)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()