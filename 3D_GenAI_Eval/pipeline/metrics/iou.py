import numpy as np
from tqdm import tqdm
import trimesh
from .base import BaseMetric
from ..constants import DEFAULT_VOXEL_PITCH, IOU_THRESHOLDS


class IoUBoolMetric(BaseMetric):
    def compute(self):
        if not self.mesh_gt.is_volume or not self.mesh_ai.is_volume:
            return 0.0
        try:
            intersection_mesh = trimesh.boolean.intersection([self.mesh_gt, self.mesh_ai])
        except ValueError:
            return 0.0
        if intersection_mesh is None:
            return 0.0
        inter_vol = intersection_mesh.volume
        union_vol = self.mesh_gt.volume + self.mesh_ai.volume - inter_vol
        return inter_vol / union_vol if union_vol > 0 else 0.0

    def get_class(self, score):
        return super().get_class(score, IOU_THRESHOLDS, reverse=False)


class IoUVoxelMetric(BaseMetric):
    def __init__(self, mesh_gt, mesh_ai, pitch=DEFAULT_VOXEL_PITCH):
        super().__init__(mesh_gt, mesh_ai)
        self.pitch = pitch

    def compute(self):
        lower = np.minimum(self.mesh_gt.bounds[0], self.mesh_ai.bounds[0])
        upper = np.maximum(self.mesh_gt.bounds[1], self.mesh_ai.bounds[1])
        xs = np.arange(lower[0], upper[0], self.pitch)
        ys = np.arange(lower[1], upper[1], self.pitch)
        zs = np.arange(lower[2], upper[2], self.pitch)

        total_intersection = 0
        total_union = 0

        for z in tqdm(zs, desc="Processing Z slices"):
            X, Y = np.meshgrid(xs, ys, indexing='ij')
            Z = np.full_like(X, z)
            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            inside_gt = self.mesh_gt.contains(points)
            inside_ai = self.mesh_ai.contains(points)
            total_intersection += np.logical_and(inside_gt, inside_ai).sum()
            total_union += np.logical_or(inside_gt, inside_ai).sum()

        return total_intersection / total_union if total_union > 0 else 0.0

    def get_class(self, score):
        return super().get_class(score, IOU_THRESHOLDS, reverse=False)
