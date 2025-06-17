import numpy as np
import math
from tqdm import tqdm
import trimesh
from .base import BaseMetric
from ..config import IOU_THRESHOLDS, VOXEL_GRID_RES, MIN_VOLUME_THRESHOLD

class IoUBoolMetric(BaseMetric):
    def compute(self, visualize=False):
        # If either mesh isn’t a true volume, we return 0.0
        if not self.mesh_gt.is_volume or not self.mesh_ai.is_volume:
            return 0.0
        try:
            intersection_mesh = trimesh.boolean.intersection([self.mesh_gt, self.mesh_ai])
        except Exception:
            return 0.0
        if intersection_mesh is None:
            return 0.0

        inter_vol = intersection_mesh.volume
        union_vol = self.mesh_gt.volume + self.mesh_ai.volume - inter_vol
        return inter_vol / union_vol if union_vol > 0 else 0.0

    def get_class(self, score):
        return super().get_class(score, IOU_THRESHOLDS, reverse=False)


class IoUVoxelMetric(BaseMetric):
    def __init__(self, mesh_gt, mesh_ai, slice_batch_size=4):
        super().__init__(mesh_gt, mesh_ai)
        self.slice_batch_size = slice_batch_size

    def compute(self, visualize=False):
        # 1) Compute a characteristic length from volume (or convex hull)
        if self.mesh_gt.is_volume and self.mesh_gt.volume > MIN_VOLUME_THRESHOLD:
            vol = self.mesh_gt.volume
        else:
            vol = self.mesh_gt.convex_hull.volume
        length = vol ** (1.0 / 3.0)   # mm

        # 2) Derive a voxel pitch so that we get ~VOXEL_GRID_RES voxels per diagonal
        pitch = length / VOXEL_GRID_RES
        print(f"Voxel pitch: {length / VOXEL_GRID_RES}")

        # 3) Build our 3D sampling grid
        lower = np.minimum(self.mesh_gt.bounds[0], self.mesh_ai.bounds[0])
        upper = np.maximum(self.mesh_gt.bounds[1], self.mesh_ai.bounds[1])
        xs = np.arange(lower[0], upper[0], pitch)
        ys = np.arange(lower[1], upper[1], pitch)
        zs = np.arange(lower[2], upper[2], pitch)

        total_intersection = 0
        total_union = 0

        # 4) Sweep across Z‐slices in batches *with* a tqdm bar
        n_batches = math.ceil(len(zs) / self.slice_batch_size)
        for z_start in tqdm(
            range(0, len(zs), self.slice_batch_size),
            total=n_batches,
            desc="Processing Z slices",
            unit="batch"
        ):
            z_batch = zs[z_start : z_start + self.slice_batch_size]
            X, Y = np.meshgrid(xs, ys, indexing='ij')

            for z in z_batch:
                Z = np.full_like(X, z)
                pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

                inside_gt = self.mesh_gt.contains(pts)
                inside_ai = self.mesh_ai.contains(pts)

                total_intersection += np.logical_and(inside_gt, inside_ai).sum()
                total_union        += np.logical_or(inside_gt, inside_ai).sum()

        return total_intersection / total_union if total_union > 0 else 0.0

    def get_class(self, score):
        return super().get_class(score, IOU_THRESHOLDS, reverse=False)
