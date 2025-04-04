import os
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm
from sklearn.decomposition import PCA
import random

class MeshEvaluator:
    def __init__(self, path_obj, path_ply):
        """
        Initialize with file paths for the AI-generated OBJ and ground truth PLY.
        """
        self.path_obj = path_obj
        self.path_ply = path_ply
        self.mesh_ai = None  # AI-generated mesh (trimesh object)
        self.mesh_gt = None  # Ground truth mesh (trimesh object)
        self.scale_factor = None

    def print_file_info(self):
        """
        Print basic file sizes.
        """
        print("=== FILE INFO ===")
        print(f"OBJ file size: {os.path.getsize(self.path_obj)} bytes")
        print(f"PLY file size: {os.path.getsize(self.path_ply)} bytes")

    def load_meshes(self):
        """
        Load the meshes from the provided file paths.
        """
        mesh_obj = trimesh.load(self.path_obj)  # AI-generated
        mesh_ply = trimesh.load(self.path_ply)  # Ground truth

        self.mesh_ai = mesh_obj
        self.mesh_gt = mesh_ply

    def print_raw_properties(self):
        """
        Print initial properties (bounding boxes and volumes) of the meshes.
        """
        print("\n=== RAW MESH PROPERTIES ===")
        print(f"OBJ bounding box extents (raw): {self.mesh_ai.bounding_box.extents}")
        print(f"PLY bounding box extents (raw): {self.mesh_gt.bounding_box.extents}")
        print(f"OBJ volume (raw): {self.mesh_ai.volume}")
        print(f"PLY volume (raw): {self.mesh_gt.volume}")

    def print_debug_info_before_transform(self):
        """
        Print debug information before any transforms.
        """
        print("\n=== BEFORE ANY TRANSFORMS ===")
        print(f"GT bounding box extents: {self.mesh_gt.bounding_box.extents}")
        print(f"AI bounding box extents: {self.mesh_ai.bounding_box.extents}")
        print(f"GT volume: {self.mesh_gt.volume}")
        print(f"AI volume: {self.mesh_ai.volume}")

    def center_meshes(self):
        """
        Translate each mesh by its center of mass so that they are centered at the origin.
        """
        self.mesh_gt.apply_translation(-self.mesh_gt.center_mass)
        self.mesh_ai.apply_translation(-self.mesh_ai.center_mass)

    def offset_ai(self, offset=[50, 0, 0]):
        """
        Offset the AI mesh for visualization purposes.
        """
        self.mesh_ai.apply_translation(offset)

    def visualize_meshes(self, title="Visualization"):
        """
        Sample point clouds from each mesh, color them differently, and visualize with Open3D.
        """
        points_gt = np.array(self.mesh_gt.sample(5000))
        points_ai = np.array(self.mesh_ai.sample(5000))

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points_gt)
        pcd_gt.paint_uniform_color([1, 0, 0])  # Red

        pcd_ai = o3d.geometry.PointCloud()
        pcd_ai.points = o3d.utility.Vector3dVector(points_ai)
        pcd_ai.paint_uniform_color([0, 1, 0])  # Green

        print(f"\n=== VISUALIZING {title.upper()} ===")
        o3d.visualization.draw_geometries([pcd_gt, pcd_ai], window_name=title)

    def print_debug_info_after_scaling(self):
        """
        Print properties of the meshes after scaling.
        """
        print("\n=== AFTER SCALING ===")
        print(f"GT bounding box extents: {self.mesh_gt.bounding_box.extents}")
        print(f"AI bounding box extents: {self.mesh_ai.bounding_box.extents}")
        print(f"GT volume: {self.mesh_gt.volume}")
        if self.scale_factor:
            print(f"AI volume: {self.mesh_ai.volume}")
        else:
            print("AI volume: Not scaled (skipped)")

    # ===========================
    # SAFE SCALING WITH HULL
    # ===========================
    def apply_safe_scaling(self):
        """
        Attempt volume-based scaling if both meshes have valid volumes.
        Otherwise, use convex hull volumes to get an approximate scale factor.
        If that fails, fall back to bounding-box scaling.
        """
        print("\n=== ATTEMPTING SAFE SCALING (Volume or Convex Hull) ===")

        # Check if both are watertight with nonzero volume
        if self.mesh_gt.is_volume and self.mesh_gt.volume > 0 and \
           self.mesh_ai.is_volume and self.mesh_ai.volume > 0:
            print("Both meshes have valid volumes. Using direct volume-based scaling.")
            self.__apply_direct_volume_scaling()
            return

        # Otherwise, attempt convex hull approach
        print("WARNING: At least one mesh not watertight or zero volume. Trying convex hull volumes.")
        hull_gt = self.mesh_gt.convex_hull
        hull_ai = self.mesh_ai.convex_hull
        vol_gt = hull_gt.volume
        vol_ai = hull_ai.volume
        print(f"GT hull volume: {vol_gt}, AI hull volume: {vol_ai}")

        if vol_gt > 1e-6 and vol_ai > 1e-6:
            self.__apply_convex_hull_volume_scaling(vol_gt, vol_ai)
        else:
            print("Convex hull volume is zero or invalid. Falling back to bounding-box diagonal scaling.")
            self.__apply_bounding_box_scaling()

    def __apply_direct_volume_scaling(self):
        """Scale AI mesh so its volume matches GT volume (both are watertight)."""
        vol_gt = self.mesh_gt.volume
        vol_ai = self.mesh_ai.volume
        self.mesh_ai.apply_translation([-50, 0, 0])
        scale_factor = (vol_gt / vol_ai) ** (1/3)
        self.scale_factor = scale_factor
        print(f"Direct volume-based scale factor: {scale_factor:.4f}")
        self.mesh_ai.apply_scale(scale_factor)
        self.mesh_ai.apply_translation([50, 0, 0])
        print("Direct volume-based scaling applied successfully.")

    def __apply_convex_hull_volume_scaling(self, vol_gt, vol_ai):
        """
        Scale AI mesh so that the AI hull volume matches the GT hull volume.
        Then apply that scale to the original AI mesh.
        """
        print("Using convex hull volumes to compute scale factor.")
        self.mesh_ai.apply_translation([-50, 0, 0])
        scale_factor = (vol_gt / vol_ai) ** (1/3)
        self.scale_factor = scale_factor
        print(f"Convex hull volume scale factor: {scale_factor:.4f}")
        self.mesh_ai.apply_scale(scale_factor)
        self.mesh_ai.apply_translation([50, 0, 0])
        print("Convex hull volume-based scaling applied successfully.")

    def __apply_bounding_box_scaling(self):
        """Fallback approach: scale AI to match bounding box diagonal of GT."""
        print("Using bounding box diagonal scaling.")
        bb_gt = self.mesh_gt.bounding_box.extents
        bb_ai = self.mesh_ai.bounding_box.extents
        diag_gt = np.linalg.norm(bb_gt)
        diag_ai = np.linalg.norm(bb_ai)

        if diag_ai < 1e-6:
            print("ERROR: AI bounding box diagonal is zero—cannot scale.")
            return

        self.mesh_ai.apply_translation([-50, 0, 0])
        scale_factor = diag_gt / diag_ai
        self.scale_factor = scale_factor
        print(f"Bounding box diagonal scale factor: {scale_factor:.4f}")
        self.mesh_ai.apply_scale(scale_factor)
        self.mesh_ai.apply_translation([50, 0, 0])
        print("Bounding box diagonal scaling applied successfully.")

    # ===========================
    # ICP + RANSAC
    # ===========================
    def apply_icp_alignment(self, num_samples=5000, threshold=10.0):
        points_gt = np.array(self.mesh_gt.sample(num_samples))
        points_ai = np.array(self.mesh_ai.sample(num_samples))

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points_gt)
        pcd_ai = o3d.geometry.PointCloud()
        pcd_ai.points = o3d.utility.Vector3dVector(points_ai)

        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_ai, pcd_gt, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        print("\n=== ICP RESULTS ===")
        print("Fitness:", reg_p2p.fitness)
        print("Inlier RMSE:", reg_p2p.inlier_rmse)
        print("Transformation:\n", reg_p2p.transformation)

        self.mesh_ai.apply_transform(reg_p2p.transformation)
        print("Applied ICP transformation to AI mesh.")

    def convert_to_open3d_full(self, num_samples=5000):
        def trimesh_to_open3d(trimesh_mesh):
            points = np.array(trimesh_mesh.sample(num_samples))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
            return pcd
        return trimesh_to_open3d(self.mesh_gt), trimesh_to_open3d(self.mesh_ai)

    def apply_ransac_global_registration(self, voxel_size=5.0):
        pcd_gt, pcd_ai = self.convert_to_open3d_full(num_samples=5000)

        def compute_fpfh(pcd, voxel_size):
            radius_feature = voxel_size * 5
            return o3d.pipelines.registration.compute_fpfh_feature(
                pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
            )

        src_fpfh = compute_fpfh(pcd_ai, voxel_size)
        tgt_fpfh = compute_fpfh(pcd_gt, voxel_size)
        distance_threshold = voxel_size * 1.5

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_ai, pcd_gt, src_fpfh, tgt_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        print("\n=== RANSAC RESULTS ===")
        print("Initial transformation:\n", result_ransac.transformation)
        self.mesh_ai.apply_transform(result_ransac.transformation)

    def apply_multiscale_icp(self, coarse_thresh=20.0, fine_thresh=5.0):
        pcd_gt, pcd_ai = self.convert_to_open3d_full(num_samples=10000)

        # Coarse ICP
        result_coarse = o3d.pipelines.registration.registration_icp(
            pcd_ai, pcd_gt, coarse_thresh, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        # Fine ICP
        result_fine = o3d.pipelines.registration.registration_icp(
            pcd_ai, pcd_gt, fine_thresh, result_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        print("\n=== ICP REFINEMENT RESULTS ===")
        print("Final transformation:\n", result_fine.transformation)
        self.mesh_ai.apply_transform(result_fine.transformation)

    def fix_if_upside_down_cat(self):
        # For example, compare bounding box corners
        gt_min, gt_max = self.mesh_gt.bounds
        ai_min, ai_max = self.mesh_ai.bounds

        # If AI's top is near AI's bottom or well below GT's top, assume it's inverted
        if ai_max[2] < gt_min[2]:
            # 180° flip around the X-axis (or Y-axis, depending on how your cat is oriented)
            R_flip = np.array([[1,  0,  0, 0],
                            [0, -1,  0, 0],
                            [0,  0, -1, 0],
                            [0,  0,  0, 1]])
            self.mesh_ai.apply_transform(R_flip)
            print("Post-flip check: Cat was upside down. Applied 180° rotation.")



class IoUEvaluator:
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai

    def compute_boolean_iou(self):
        if not self.mesh_gt.is_volume or not self.mesh_ai.is_volume:
            print("WARNING: One or both meshes are not watertight. Boolean IoU cannot be computed reliably.")
            return 0.0
        try:
            intersection_mesh = trimesh.boolean.intersection([self.mesh_gt, self.mesh_ai])
        except ValueError as e:
            print("Boolean intersection error:", e)
            return 0.0

        if intersection_mesh is None:
            print("Boolean intersection returned None.")
            intersection_volume = 0.0
        else:
            intersection_volume = intersection_mesh.volume

        vol_gt = self.mesh_gt.volume
        vol_ai = self.mesh_ai.volume
        union_volume = vol_gt + vol_ai - intersection_volume
        if union_volume == 0:
            return 0.0
        return intersection_volume / union_volume

    def compute_voxel_iou_progress(self, pitch=5.0):
        lower = np.minimum(self.mesh_gt.bounds[0], self.mesh_ai.bounds[0])
        upper = np.maximum(self.mesh_gt.bounds[1], self.mesh_ai.bounds[1])
        
        xs = np.arange(lower[0], upper[0], pitch)
        ys = np.arange(lower[1], upper[1], pitch)
        zs = np.arange(lower[2], upper[2], pitch)

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

        if total_union > 0:
            return total_intersection / total_union
        else:
            return 0.0


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    path_obj = 'genAI_models/obj_09.obj'
    path_ply = 'models/obj_09.ply'

    evaluator = MeshEvaluator(path_obj, path_ply)
    evaluator.print_file_info()
    evaluator.load_meshes()
    evaluator.print_raw_properties()
    evaluator.print_debug_info_before_transform()

    # 1. Center
    evaluator.center_meshes()

    # 2. Offset AI for visualization
    evaluator.offset_ai()

    # Visualize BEFORE scaling
    evaluator.visualize_meshes(title="Before Scaling")

    # 3. Safe scaling (volume-based, hull-based, or bounding-box)
    evaluator.apply_safe_scaling()
    evaluator.print_debug_info_after_scaling()

    # Visualize AFTER scaling
    evaluator.visualize_meshes(title="After Scaling")

    # 4. Remove offset
    evaluator.mesh_ai.apply_translation([-50, 0, 0])

    # 5. RANSAC + Multi-scale ICP
    evaluator.apply_ransac_global_registration(voxel_size=5.0)
    evaluator.apply_multiscale_icp(coarse_thresh=20.0, fine_thresh=5.0)
    #evaluator.fix_if_upside_down_cat()

    # Visualize final
    evaluator.visualize_meshes(title="After ICP")

    # 6. IoU
    iou_evaluator = IoUEvaluator(evaluator.mesh_gt, evaluator.mesh_ai)
    boolean_iou = iou_evaluator.compute_boolean_iou()
    voxel_iou = iou_evaluator.compute_voxel_iou_progress(pitch=5.5)

    print("\n=== IOU RESULTS ===")
    print(f"Boolean IoU: {boolean_iou:.4f}")
    print(f"Voxel-based IoU (with progress): {voxel_iou:.4f}")
    print("\n=== DONE ===")
