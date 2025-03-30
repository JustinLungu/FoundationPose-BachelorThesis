import numpy as np
from typing import Tuple
from ..data_loader.models import Transformation

class MetricCalculator:
    @staticmethod
    def compute_all(gt: Transformation, pred: Transformation, points: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute all metrics for a single frame pair"""
        return (
            MetricCalculator.rotation_error(gt.matrix, pred.matrix),
            MetricCalculator.translation_error(gt.matrix, pred.matrix),
            MetricCalculator.pose_error(gt.matrix, pred.matrix),
            MetricCalculator.add_error(gt.matrix, pred.matrix, points)
        )

    @staticmethod
    def rotation_error(T_gt: np.ndarray, T_pred: np.ndarray) -> float:
        """Rotation error in degrees"""
        return np.rad2deg(np.arccos(np.clip((np.trace(T_gt[:3,:3].T @ T_pred[:3,:3]) - 1)/2, -1, 1)))

    @staticmethod
    def translation_error(T_gt: np.ndarray, T_pred: np.ndarray) -> float:
        """Euclidean distance between translations (meters)"""
        return np.linalg.norm(T_gt[:3,3] - T_pred[:3,3])

    @staticmethod
    def pose_error(T_gt: np.ndarray, T_pred: np.ndarray) -> float:
        """Frobenius norm of matrix difference"""
        return np.linalg.norm(T_gt - T_pred, 'fro')

    @staticmethod
    def add_error(T_gt: np.ndarray, T_pred: np.ndarray, points: np.ndarray) -> float:
        """Average Distance (ADD) in meters"""
        pts_gt = (T_gt[:3,:3] @ points.T).T + T_gt[:3,3]
        pts_pred = (T_pred[:3,:3] @ points.T).T + T_pred[:3,3]
        return np.mean(np.linalg.norm(pts_gt - pts_pred, axis=1))