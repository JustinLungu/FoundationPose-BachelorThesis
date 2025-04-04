import numpy as np
import trimesh
import open3d as o3d

# 1. LOAD MESHES
mesh_gt = trimesh.load('models/obj_01.ply')  # Ground truth
mesh_ai = trimesh.load('genAI_models/obj_01.obj')  # AI-generated

# 2. PRINT DEBUG INFO (bounding boxes & volumes before any transform)
print("GT bounding box extents (before):", mesh_gt.bounding_box.extents)
print("AI bounding box extents (before):", mesh_ai.bounding_box.extents)
print("GT volume (before):", mesh_gt.volume)
print("AI volume (before):", mesh_ai.volume)

# 3. CENTER AT ORIGIN (translate by center of mass)
mesh_gt.apply_translation(-mesh_gt.center_mass)
mesh_ai.apply_translation(-mesh_ai.center_mass)

# 4. TRANSLATE AI MESH SO IT'S VISIBLY OFFSET (to see it before scaling)
mesh_ai.apply_translation([50, 0, 0])

# 5. SAMPLE POINT CLOUDS (BEFORE SCALING)
points_gt_before = np.array(mesh_gt.sample(5000))
points_ai_before = np.array(mesh_ai.sample(5000))

# 6. CREATE OPEN3D POINT CLOUDS (with different colors) FOR "BEFORE SCALING"
pcd_gt_before = o3d.geometry.PointCloud()
pcd_gt_before.points = o3d.utility.Vector3dVector(points_gt_before)
pcd_gt_before.paint_uniform_color([1, 0, 0])  # Red

pcd_ai_before = o3d.geometry.PointCloud()
pcd_ai_before.points = o3d.utility.Vector3dVector(points_ai_before)
pcd_ai_before.paint_uniform_color([0, 1, 0])  # Green

# 7. VISUALIZE "BEFORE SCALING"
print("Visualizing before volume scaling...")
o3d.visualization.draw_geometries([pcd_gt_before, pcd_ai_before], window_name="Before Scaling")

# 8. APPLY VOLUME-BASED SCALING TO AI MESH
vol_gt = mesh_gt.volume
vol_ai = mesh_ai.volume

if vol_ai > 0 and vol_gt > 0:
    scale_factor = (vol_gt / vol_ai) ** (1/3)
    # IMPORTANT: remove the +50 translation first so scaling happens around the origin
    mesh_ai.apply_translation([-50, 0, 0])  
    mesh_ai.apply_scale(scale_factor)
    # Re-apply the translation to keep it offset for visualization
    mesh_ai.apply_translation([50, 0, 0])
    print(f"Applied scale factor to AI mesh: {scale_factor}")
else:
    print("WARNING: One of the meshes has zero volume; skipping volume-based scaling.")

# 9. SAMPLE POINT CLOUDS (AFTER SCALING)
points_gt_after = np.array(mesh_gt.sample(5000))  # GT is unchanged
points_ai_after = np.array(mesh_ai.sample(5000))  # AI is now scaled

# 10. CREATE OPEN3D POINT CLOUDS FOR "AFTER SCALING"
pcd_gt_after = o3d.geometry.PointCloud()
pcd_gt_after.points = o3d.utility.Vector3dVector(points_gt_after)
pcd_gt_after.paint_uniform_color([1, 0, 0])  # Red

pcd_ai_after = o3d.geometry.PointCloud()
pcd_ai_after.points = o3d.utility.Vector3dVector(points_ai_after)
pcd_ai_after.paint_uniform_color([0, 1, 0])  # Green

# 11. VISUALIZE "AFTER SCALING"
print("Visualizing after volume scaling...")
o3d.visualization.draw_geometries([pcd_gt_after, pcd_ai_after], window_name="After Scaling")
