import trimesh
import numpy as np

path_ply = 'models/obj_01.ply'   # GT
path_obj = 'genAI_models/obj_01.obj'   # AI

mesh_gt = trimesh.load(path_ply)
mesh_ai = trimesh.load(path_obj)


# Translate ground truth to origin
mesh_gt.apply_translation(-mesh_gt.center_mass)

# Translate AI mesh to origin
mesh_ai.apply_translation(-mesh_ai.center_mass)

bb_gt = mesh_gt.bounding_box.extents
bb_ai = mesh_ai.bounding_box.extents

print("AI bounding box extents before scaling:", mesh_ai.bounding_box.extents)
print("AI volume before scaling:", mesh_ai.volume)

# Example: match the maximum extent
scale_factor = bb_gt.max() / bb_ai.max()
mesh_ai.apply_scale(scale_factor)


print("GT bounding box extents:", mesh_gt.bounding_box.extents)
print("AI bounding box extents:", mesh_ai.bounding_box.extents)
print("GT volume:", mesh_gt.volume)
print("AI volume:", mesh_ai.volume)
