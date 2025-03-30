import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

class Plotter:
    def __init__(self):
        self.figsize = (10, 8)
        self.dpi = 150

    def plot_metrics(self, metrics: Dict[str, List[float]], save_path: Path):
        """Generate all diagnostic plots"""
        self._plot_trends(metrics, save_path / "trends.png")
        self._plot_distributions(metrics, save_path / "distributions.png")

    def _plot_trends(self, metrics: Dict[str, List[float]], save_path: Path):
        fig, axs = plt.subplots(4, 1, figsize=self.figsize)
        frames = np.arange(len(metrics['rotation']))
        
        self._plot_metric(axs[0], frames, metrics['rotation'], 
                         "Rotation Error (deg)", [5, 10])
        self._plot_metric(axs[1], frames, metrics['translation'], 
                         "Translation Error (m)", [0.01, 0.05])
        self._plot_metric(axs[2], frames, metrics['pose'], 
                         "Pose Error", [0.1, 0.3])
        self._plot_metric(axs[3], frames, metrics['add'], 
                         "ADD (m)", [0.01, 0.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

    def _plot_distributions(self, metrics: Dict[str, List[float]], save_path: Path):
        fig, axs = plt.subplots(4, 1, figsize=self.figsize)
        
        self._plot_hist(axs[0], metrics['rotation'], "Rotation Errors")
        self._plot_hist(axs[1], metrics['translation'], "Translation Errors")
        self._plot_hist(axs[2], metrics['pose'], "Pose Errors")
        self._plot_hist(axs[3], metrics['add'], "ADD Errors")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

    def _plot_metric(self, ax, x, y, title, thresholds):
        ax.plot(x, y, 'b-')
        ax.set_title(title)
        for t in thresholds:
            ax.axhline(t, color='r' if t == thresholds[-1] else 'g', linestyle='--')

    def _plot_hist(self, ax, data, title, bins=50):
        ax.hist(data, bins=bins, alpha=0.7)
        ax.set_title(title)