import numpy as np
from constants import DEFAULT_OFFSET, MIN_VOLUME_THRESHOLD, ZERO_TOLERANCE

class MeshPreprocessor:
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai
        self.scale_factor = None

    def center(self):
        self.mesh_gt.apply_translation(-self.mesh_gt.center_mass)
        self.mesh_ai.apply_translation(-self.mesh_ai.center_mass)

    def safe_scaling(self):
        if self.mesh_gt.is_volume and self.mesh_gt.volume > MIN_VOLUME_THRESHOLD and \
           self.mesh_ai.is_volume and self.mesh_ai.volume > MIN_VOLUME_THRESHOLD:
            return self._scale_by_volume(self.mesh_gt.volume, self.mesh_ai.volume)
        else:
            vol_gt = self.mesh_gt.convex_hull.volume
            vol_ai = self.mesh_ai.convex_hull.volume
            if vol_gt > MIN_VOLUME_THRESHOLD and vol_ai > MIN_VOLUME_THRESHOLD:
                return self._scale_by_volume(vol_gt, vol_ai)
            return self._scale_by_bounding_box()

    def _scale_by_volume(self, vol_gt, vol_ai):
        scale_factor = (vol_gt / vol_ai) ** (1/3)
        self.scale_factor = scale_factor
        self.mesh_ai.apply_scale(scale_factor)
        return scale_factor

    def _scale_by_bounding_box(self):
        ext_gt = self.mesh_gt.bounding_box.extents
        ext_ai = self.mesh_ai.bounding_box.extents
        diag_gt = np.linalg.norm(ext_gt)
        diag_ai = np.linalg.norm(ext_ai)
        if diag_ai < ZERO_TOLERANCE:
            return None
        scale_factor = diag_gt / diag_ai
        self.scale_factor = scale_factor
        self.mesh_ai.apply_scale(scale_factor)
        return scale_factor

    # Optional: PCA alignment (not used in current pipeline)
    def pca_align(self):
        def get_basis(points):
            centered = points - points.mean(0)
            _, _, Vt = np.linalg.svd(centered)
            return Vt.T

        ai_points = np.array(self.mesh_ai.sample(5000))
        gt_points = np.array(self.mesh_gt.sample(5000))
        R = get_basis(gt_points) @ get_basis(ai_points).T

        T = np.eye(4)
        T[:3, :3] = R
        self.mesh_ai.apply_transform(T)
