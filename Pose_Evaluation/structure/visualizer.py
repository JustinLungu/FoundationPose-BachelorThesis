import matplotlib.pyplot as plt
import numpy as np

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

        self._scatter_plot(axs[0], self.rotation_errors, rot_th, "Rotation Error Outliers", "Degrees")
        self._scatter_plot(axs[1], self.translation_errors, trans_th, "Translation Error Outliers", "Meters")
        self._scatter_plot(axs[2], self.pose_errors, pose_th, "Pose Error Outliers", "Error")
        self._scatter_plot(axs[3], self.add_errors, add_th, "ADD Error Outliers", "Meters")

        plt.tight_layout()
        plt.savefig("plots/error_outliers.png")
        plt.close()

    def plot_trends(self):
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))
        
        self._trend_plot(axs[0], self.rotation_errors, [5, 10], "Rotation Error Over Frames", "Degrees")
        self._trend_plot(axs[1], self.translation_errors, [0.01, 0.05], "Translation Error Over Frames", "Meters")
        self._trend_plot(axs[2], self.pose_errors, [0.1, 0.3], "Pose Error Over Frames", "Error")
        self._trend_plot(axs[3], self.add_errors, [0.01, 0.05], "ADD Error Over Frames", "Meters")

        plt.tight_layout()
        plt.savefig("plots/error_trends.png")
        plt.close()

    def plot_distributions(self):
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        self._histogram(axs[0], self.rotation_errors, [5, 10], "Rotation Error Distribution", "Degrees")
        self._histogram(axs[1], self.translation_errors, [0.01, 0.05], "Translation Error Distribution", "Meters")
        self._histogram(axs[2], self.pose_errors, [0.1, 0.3], "Pose Error Distribution", "Error")
        self._histogram(axs[3], self.add_errors, [0.01, 0.05], "ADD Error Distribution", "Meters")

        plt.tight_layout()
        plt.savefig("plots/error_distributions.png")
        plt.close()

    def _scatter_plot(self, ax, errors, threshold, title, ylabel):
        ax.scatter(self.frames, errors, alpha=0.5)
        ax.scatter([i for i in self.frames if errors[i] > threshold],
                   [errors[i] for i in self.frames if errors[i] > threshold],
                   color='red', label=f"Outlier (>{threshold})")
        ax.set_title(title)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel(ylabel)
        ax.legend()

    def _trend_plot(self, ax, errors, thresholds, title, ylabel):
        ax.plot(self.frames, errors)
        ax.axhline(thresholds[0], color='green', linestyle='--', label=f'Good (<{thresholds[0]})')
        ax.axhline(thresholds[1], color='red', linestyle='--', label=f'Bad (>{thresholds[1]})')
        ax.set_title(title)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel(ylabel)
        ax.legend()

    def _histogram(self, ax, errors, thresholds, title, ylabel):
        ax.hist(errors, bins=50, alpha=0.7)
        ax.axvline(thresholds[0], color='green', linestyle='--', label=f'Good (<{thresholds[0]})')
        ax.axvline(thresholds[1], color='red', linestyle='--', label=f'Bad (>{thresholds[1]})')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
