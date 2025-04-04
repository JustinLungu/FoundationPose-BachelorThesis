import os
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm
from sklearn.decomposition import PCA

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
        mesh_ply = trimesh.load(self.path_ply)    # Ground truth

        # Assign to instance variables:
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
        pcd_gt.paint_uniform_color([1, 0, 0])  # Red for ground truth

        pcd_ai = o3d.geometry.PointCloud()
        pcd_ai.points = o3d.utility.Vector3dVector(points_ai)
        pcd_ai.paint_uniform_color([0, 1, 0])  # Green for AI-generated

        print(f"\n=== VISUALIZING {title.upper()} ===")
        o3d.visualization.draw_geometries([pcd_gt, pcd_ai], window_name=title)

    def apply_volume_scaling(self):
        """
        Compute the uniform scaling factor to match volumes, then scale the AI mesh.
        To scale around the origin, the pre-applied offset is temporarily removed.
        """
        vol_gt = self.mesh_gt.volume
        vol_ai = self.mesh_ai.volume

        print("\n=== APPLYING VOLUME-BASED SCALING TO AI MESH ===")
        if vol_ai > 0 and vol_gt > 0:
            self.scale_factor = (vol_gt / vol_ai) ** (1/3)
            print(f"Calculated scale factor: {self.scale_factor:.4f}")

            # Remove the offset (assumed to be [50, 0, 0]) so scaling happens around the origin
            self.mesh_ai.apply_translation([-50, 0, 0])
            self.mesh_ai.apply_scale(self.scale_factor)
            # Re-apply the offset for visualization
            self.mesh_ai.apply_translation([50, 0, 0])
            print("Scaling applied successfully.")
        else:
            self.scale_factor = None
            print("WARNING: One mesh has zero volume—skipping volume-based scaling.")

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

    def apply_pca_orientation_alignment(self):
        """
        Reorient both meshes using PCA so that their 'up' direction (the axis of smallest variance)
        aligns with the global Z axis. This helps ensure that the models have the same overall orientation.
        """
        def pca_alignment(mesh):
            # Sample a set of points from the mesh.
            points = np.array(mesh.sample(10000))
            pca = PCA(n_components=3)
            pca.fit(points)
            # The PCA components are sorted by variance (largest first).
            # We assume the component with the smallest variance (last one) is the 'up' direction.
            up = pca.components_[-1]
            # Force the up vector to point in the positive Z direction.
            if up[2] < 0:
                up = -up
            # Compute the rotation that aligns 'up' with [0,0,1].
            # Use the cross product between up and [0,0,1] as the rotation axis.
            axis = np.cross(up, [0,0,1])
            norm_axis = np.linalg.norm(axis)
            if norm_axis < 1e-6:
                return np.eye(3)
            axis = axis / norm_axis
            angle = np.arccos(np.clip(np.dot(up, [0,0,1]), -1.0, 1.0))
            # Rodrigues formula for rotation matrix
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
            return R

        # Compute rotation matrices for both meshes.
        R_gt = pca_alignment(self.mesh_gt)
        R_ai = pca_alignment(self.mesh_ai)
        print("\n=== PCA ORIENTATION ALIGNMENT ===")
        print("Rotation matrix for GT mesh:\n", R_gt)
        print("Rotation matrix for AI mesh:\n", R_ai)
        # Convert to 4x4 homogeneous transformation matrices.
        T_gt = np.eye(4)
        T_gt[:3,:3] = R_gt
        T_ai = np.eye(4)
        T_ai[:3,:3] = R_ai
        # Apply the transformations.
        self.mesh_gt.apply_transform(T_gt)
        self.mesh_ai.apply_transform(T_ai)
        print("Applied PCA orientation alignment to both meshes.")

    def apply_icp_alignment(self, num_samples=5000, threshold=10.0):
        """
        Refine alignment between mesh_gt and mesh_ai using point-to-point ICP.
        
        :param num_samples: Number of points to sample from each mesh for ICP.
        :param threshold: Max correspondence distance for ICP (adjust as needed).
        """
        # 1. Convert trimesh -> numpy arrays
        points_gt = np.array(self.mesh_gt.sample(num_samples))
        points_ai = np.array(self.mesh_ai.sample(num_samples))

        # 2. Create Open3D point clouds
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points_gt)
        pcd_ai = o3d.geometry.PointCloud()
        pcd_ai.points = o3d.utility.Vector3dVector(points_ai)

        # 3. Initial guess (identity transform)
        trans_init = np.eye(4)

        # 4. Run ICP (point-to-point)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_ai, pcd_gt, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        print("\n=== ICP RESULTS ===")
        print("Fitness:", reg_p2p.fitness)
        print("Inlier RMSE:", reg_p2p.inlier_rmse)
        print("Transformation:\n", reg_p2p.transformation)

        # 5. Apply the ICP transform to mesh_ai
        self.mesh_ai.apply_transform(reg_p2p.transformation)
        print("Applied ICP transformation to AI mesh.")

    # --- Additional Alignment Methods ---

    def convert_to_open3d_full(self, num_samples=5000):
        """
        Convert Trimesh objects to Open3D point clouds.
        """
        def trimesh_to_open3d(trimesh_mesh):
            points = np.array(trimesh_mesh.sample(num_samples))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
            return pcd
        return trimesh_to_open3d(self.mesh_gt), trimesh_to_open3d(self.mesh_ai)

    def apply_ransac_global_registration(self, voxel_size=5.0):
        """
        Use RANSAC feature-based global registration for an initial rough alignment.
        """
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

        # Apply RANSAC transformation to AI mesh
        self.mesh_ai.apply_transform(result_ransac.transformation)

    def apply_multiscale_icp(self, coarse_thresh=20.0, fine_thresh=5.0):
        """
        Perform multi-scale ICP (coarse-to-fine) using point-to-plane registration.
        """
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

        # Apply final ICP transformation
        self.mesh_ai.apply_transform(result_fine.transformation)


class IoUEvaluator:
    def __init__(self, mesh_gt, mesh_ai):
        """
        Initialize with the ground truth and AI-generated meshes.
        """
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai

    def compute_boolean_iou(self):
        """
        Compute IoU using boolean mesh intersection.
        Returns IoU value or 0 if intersection fails.
        """
        # First check if both meshes are volumes/watertight:
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
        iou = intersection_volume / union_volume
        return iou

    def compute_voxel_iou_progress(self, pitch=5.0):
        """
        Compute an approximate IoU using a voxel grid.
        Processes the grid in Z-slices with a given pitch (voxel size) and shows progress.
        Increasing the pitch reduces the resolution (and memory usage).
        """
        # Determine the combined bounding box for both meshes
        lower = np.minimum(self.mesh_gt.bounds[0], self.mesh_ai.bounds[0])
        upper = np.maximum(self.mesh_gt.bounds[1], self.mesh_ai.bounds[1])
        
        xs = np.arange(lower[0], upper[0], pitch)
        ys = np.arange(lower[1], upper[1], pitch)
        zs = np.arange(lower[2], upper[2], pitch)

        total_intersection = 0
        total_union = 0

        # Process one Z-slice at a time to limit memory usage
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
    path_obj = 'genAI_models/obj_10.obj'
    path_ply = 'models/obj_10.ply'

    evaluator = MeshEvaluator(path_obj, path_ply)
    evaluator.print_file_info()
    evaluator.load_meshes()
    evaluator.print_raw_properties()
    evaluator.print_debug_info_before_transform()

    # 1. Center the meshes at the origin
    evaluator.center_meshes()

    # 2. Offset AI mesh for visualization (optional)
    evaluator.offset_ai()

    # Visualize BEFORE scaling
    evaluator.visualize_meshes(title="Before Scaling")

    # 3. Volume-based scaling
    evaluator.apply_volume_scaling()
    evaluator.print_debug_info_after_scaling()

    # Visualize AFTER scaling (still offset by +50)
    evaluator.visualize_meshes(title="After Scaling")

    # 4. Remove offset so the AI mesh is back in the same space as GT
    evaluator.mesh_ai.apply_translation([-50, 0, 0])

    # 4.5. Apply PCA orientation alignment so both meshes have the same up (e.g., head up)
    evaluator.apply_pca_orientation_alignment()

    # 5. Apply additional alignment: RANSAC global registration + Multi-scale ICP
    evaluator.apply_ransac_global_registration(voxel_size=5.0)
    evaluator.apply_multiscale_icp(coarse_thresh=20.0, fine_thresh=5.0)

    # Visualize AFTER ICP refinement (final alignment)
    evaluator.visualize_meshes(title="After ICP")

    # 6. Compute IoU
    iou_evaluator = IoUEvaluator(evaluator.mesh_gt, evaluator.mesh_ai)

    boolean_iou = iou_evaluator.compute_boolean_iou()
    voxel_iou = iou_evaluator.compute_voxel_iou_progress(pitch=5.5)

    print("\n=== IOU RESULTS ===")
    print(f"Boolean IoU: {boolean_iou:.4f}")
    print(f"Voxel-based IoU (with progress): {voxel_iou:.4f}")

    print("\n=== DONE ===")
