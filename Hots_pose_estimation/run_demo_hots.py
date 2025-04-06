import sys
import os
import numpy as np
import trimesh
import imageio
import cv2
import glob

# Add both project root and FoundationPose to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FoundationPose')))

from FoundationPose.estimater import *
from FoundationPose.datareader import *

# CONFIG FLAG — toggle this to True for HOTS-like mask-per-frame behavior
use_mask_every_frame = True

############## 1. SETUP GLOBAL PATHS ##############
base_dir = os.path.dirname(os.path.abspath(__file__))
demo_root = os.path.join(base_dir, "data", "HOTS_Processed_demo")
output_root = os.path.join(base_dir, "results", "demo_run")

os.makedirs(output_root, exist_ok=True)

############## 2. LOOP THROUGH ALL OBJECT FOLDERS ##############
object_folders = sorted(os.listdir(demo_root))

for obj_name in object_folders:
    data_root = os.path.join(demo_root, obj_name)
    mesh_file = os.path.join(data_root, "mesh", "model.obj")
    test_scene_dir = data_root
    rgb_files = glob.glob(os.path.join(data_root, "rgb", "*.png"))

    # Skip if mesh or RGB images are missing
    if not os.path.exists(mesh_file) or len(rgb_files) == 0:
        print(f"Skipping '{obj_name}' (missing mesh or RGB)")
        continue

    print(f"\nProcessing object: {obj_name}")
    debug_dir = os.path.join(output_root, obj_name)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "ob_in_cam"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "track_vis"), exist_ok=True)

    ############## 3. INITIALIZE MODEL & MESH ##############
    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=2,
        glctx=glctx
    )

    ############## 4. LOAD RGB-D FRAMES & RUN POSE ESTIMATION ##############
    reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf, per_frame_masks=True)

    for i in range(len(reader.color_files)):
        print(f'Frame {i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0 or use_mask_every_frame:
            mask = reader.get_mask(i).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=2)

        # Save pose
        np.savetxt(os.path.join(debug_dir, "ob_in_cam", f"{reader.id_strs[i]}.txt"), pose.reshape(4, 4))

        # Save visualization
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        imageio.imwrite(os.path.join(debug_dir, "track_vis", f"{reader.id_strs[i]}.png"), vis)

        # Optional display
        cv2.imshow("Prediction", vis[..., ::-1])
        cv2.waitKey(1)
