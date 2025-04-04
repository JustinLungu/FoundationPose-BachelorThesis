from abc import ABC, abstractmethod
import os
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm


class MeshLoader:
    def __init__(self, path_obj: str, path_ply: str):
        self.path_obj = path_obj
        self.path_ply = path_ply
        self.mesh_ai = None
        self.mesh_gt = None

    def load(self):
        self.mesh_ai = trimesh.load(self.path_obj)
        self.mesh_gt = trimesh.load(self.path_ply)

    def get_meshes(self):
        return self.mesh_gt, self.mesh_ai


class MeshPreprocessor:
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai
        self.scale_factor = None

    def center(self):
        self.mesh_gt.apply_translation(-self.mesh_gt.center_mass)
        self.mesh_ai.apply_translation(-self.mesh_ai.center_mass)

    def offset_ai(self, offset=[50, 0, 0]):
        self.mesh_ai.apply_translation(offset)

    def remove_offset_ai(self, offset=[-50, 0, 0]):
        self.mesh_ai.apply_translation(offset)

    def safe_scaling(self):
        if self.mesh_gt.is_volume and self.mesh_gt.volume > 0 and \
           self.mesh_ai.is_volume and self.mesh_ai.volume > 0:
            return self._scale_by_volume(self.mesh_gt.volume, self.mesh_ai.volume)
        else:
            vol_gt = self.mesh_gt.convex_hull.volume
            vol_ai = self.mesh_ai.convex_hull.volume
            if vol_gt > 1e-6 and vol_ai > 1e-6:
                return self._scale_by_volume(vol_gt, vol_ai)
            return self._scale_by_bounding_box()

    def _scale_by_volume(self, vol_gt, vol_ai):
        self.mesh_ai.apply_translation([-50, 0, 0])
        self.scale_factor = (vol_gt / vol_ai) ** (1/3)
        self.mesh_ai.apply_scale(self.scale_factor)
        self.mesh_ai.apply_translation([50, 0, 0])
        return self.scale_factor

    def _scale_by_bounding_box(self):
        ext_gt = self.mesh_gt.bounding_box.extents
        ext_ai = self.mesh_ai.bounding_box.extents
        diag_gt = np.linalg.norm(ext_gt)
        diag_ai = np.linalg.norm(ext_ai)
        if diag_ai < 1e-6:
            return None
        self.mesh_ai.apply_translation([-50, 0, 0])
        self.scale_factor = diag_gt / diag_ai
        self.mesh_ai.apply_scale(self.scale_factor)
        self.mesh_ai.apply_translation([50, 0, 0])
        return self.scale_factor


class MeshVisualizer:
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai

    def show(self, title="Visualization"):
        points_gt = np.array(self.mesh_gt.sample(5000))
        points_ai = np.array(self.mesh_ai.sample(5000))

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points_gt)
        pcd_gt.paint_uniform_color([1, 0, 0])

        pcd_ai = o3d.geometry.PointCloud()
        pcd_ai.points = o3d.utility.Vector3dVector(points_ai)
        pcd_ai.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([pcd_gt, pcd_ai], window_name=title)


class Metric(ABC):
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai

    @abstractmethod
    def compute(self):
        pass


class IoUBoolMetric(Metric):
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


class IoUVoxelMetric(Metric):
    def __init__(self, mesh_gt, mesh_ai, pitch=5.0):
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


if __name__ == "__main__":
    loader = MeshLoader('genAI_models/obj_06.obj', 'models/obj_06.ply')
    loader.load()
    mesh_gt, mesh_ai = loader.get_meshes()

    preprocessor = MeshPreprocessor(mesh_gt, mesh_ai)
    preprocessor.center()
    preprocessor.offset_ai()

    vis = MeshVisualizer(mesh_gt, mesh_ai)
    vis.show("Before Scaling")

    preprocessor.safe_scaling()
    vis.show("After Scaling")

    preprocessor.remove_offset_ai()

    bool_iou = IoUBoolMetric(mesh_gt, mesh_ai).compute()
    voxel_iou = IoUVoxelMetric(mesh_gt, mesh_ai, pitch=5.5).compute()

    print("\n=== IOU RESULTS ===")
    print(f"Boolean IoU: {bool_iou:.4f}")
    print(f"Voxel-based IoU: {voxel_iou:.4f}")