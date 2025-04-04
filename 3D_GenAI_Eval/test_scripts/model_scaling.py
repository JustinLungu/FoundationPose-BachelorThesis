import os
import numpy as np
import trimesh
import open3d as o3d

class MeshEvaluator:
    def __init__(self, path_obj, path_ply):
        """
        Initialize with paths for the AI-generated OBJ and ground truth PLY.
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

        # For clarity, assign to our instance variables:
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
        Note: To scale around the origin, the pre-applied offset is temporarily removed.
        """
        vol_gt = self.mesh_gt.volume
        vol_ai = self.mesh_ai.volume

        print("\n=== APPLYING VOLUME-BASED SCALING TO AI MESH ===")
        if vol_ai > 0 and vol_gt > 0:
            self.scale_factor = (vol_gt / vol_ai) ** (1/3)
            print(f"Calculated scale factor: {self.scale_factor:.4f}")

            # Remove the offset (assumed to be [50, 0, 0]) so scaling is about the origin.
            self.mesh_ai.apply_translation([-50, 0, 0])
            self.mesh_ai.apply_scale(self.scale_factor)
            # Re-apply the offset for visualization.
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


if __name__ == "__main__":
    # ----------------------------
    # Define file paths
    # ----------------------------
    path_obj = 'genAI_models/obj_01.obj'
    path_ply = 'models/obj_01.ply'

    # ----------------------------
    # Create an evaluator instance and run methods
    # ----------------------------
    evaluator = MeshEvaluator(path_obj, path_ply)

    # Print file information
    evaluator.print_file_info()

    # Load meshes and print raw properties
    evaluator.load_meshes()
    evaluator.print_raw_properties()

    # Print debug info before transformations
    evaluator.print_debug_info_before_transform()

    # Center the meshes at the origin
    evaluator.center_meshes()

    # Offset the AI mesh so it’s visible separately
    evaluator.offset_ai()

    # Visualize BEFORE scaling
    evaluator.visualize_meshes(title="Before Scaling")

    # Apply volume-based scaling to the AI mesh
    evaluator.apply_volume_scaling()

    # Print debug info after scaling
    evaluator.print_debug_info_after_scaling()

    # Visualize AFTER scaling
    evaluator.visualize_meshes(title="After Scaling")

    print("\n=== DONE ===")
