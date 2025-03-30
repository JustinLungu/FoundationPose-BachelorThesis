import numpy as np
from typing import List, Dict
from ..data_loader.models import PointCloud, Transformation
from .metrics import MetricCalculator

class PoseEvaluator:
    def __init__(self, 
                 gt_transforms: List[Transformation],
                 pred_transforms: List[Transformation],
                 point_cloud: PointCloud):
        self.gt = gt_transforms
        self.pred = pred_transforms
        self.points = point_cloud
        self._validate()

    def _validate(self):
        if len(self.gt) != len(self.pred):
            raise ValueError("Ground truth and prediction count mismatch")
        if len(self.points.points) == 0:
            raise ValueError("Point cloud is empty")

    def evaluate(self) -> Dict[str, np.ndarray]:
        """Compute all metrics for frame pairs"""
        results = {
            'rotation': [],
            'translation': [],
            'pose': [],
            'add': []
        }
        
        for gt, pred in zip(self.gt, self.pred):
            rot_err, trans_err, pose_err, add_err = MetricCalculator.compute_all(
                gt, pred, self.points.points
            )
            results['rotation'].append(rot_err)
            results['translation'].append(trans_err)
            results['pose'].append(pose_err)
            results['add'].append(add_err)
            
        return results