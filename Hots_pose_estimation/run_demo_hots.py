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

# === CONSTANTS ===
USE_MASK_EVERY_FRAME = True
DEBUG_LEVEL = 2
ITERATION_REGISTER = 5
ITERATION_TRACK = 2
AXIS_SCALE = 0.1
AXIS_THICKNESS = 3
TRANSPARENCY = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_ROOT = os.path.join(BASE_DIR, "data", "HOTS_Processed_demo")
OUTPUT_ROOT = os.path.join(BASE_DIR, "results", "demo_run")

# === INITIALIZE OUTPUT FOLDER ===
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# === LOOP THROUGH ALL OBJECT FOLDERS ===
object_folders = sorted(os.listdir(DEMO_ROOT))

for obj_name in object_folders:
    data_root = os.path.join(DEMO_ROOT, obj_name)
    mesh_file = os.path.join(data_root, "mesh", "model.obj")
    test_scene_dir = data_root
    rgb_files = glob.glob(os.path.join(data_root, "rgb", "*.png"))

    # Skip if mesh or RGB images are missing
    if not os.path.exists(mesh_file) or len(rgb_files) == 0:
        print(f"Skipping '{obj_name}' (missing mesh or RGB)")
        continue

    print(f"\nProcessing object: {obj_name}")
    debug_dir = os.path.join(OUTPUT_ROOT, obj_name)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "ob_in_cam"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "track_vis"), exist_ok=True)

    # === INITIALIZE MODEL & MESH ===
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
        debug=DEBUG_LEVEL,
        glctx=glctx
    )

    # === LOAD RGB-D FRAMES & RUN POSE ESTIMATION ===
    reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf, per_frame_masks=True)

    for i in range(len(reader.color_files)):
        print(f'Frame {i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0 or USE_MASK_EVERY_FRAME:
            mask = reader.get_mask(i).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=ITERATION_REGISTER)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=ITERATION_TRACK)

        # Save pose
        np.savetxt(os.path.join(debug_dir, "ob_in_cam", f"{reader.id_strs[i]}.txt"), pose.reshape(4, 4))

        # Save visualization
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=AXIS_SCALE, K=reader.K, thickness=AXIS_THICKNESS, transparency=TRANSPARENCY, is_input_rgb=True)
        imageio.imwrite(os.path.join(debug_dir, "track_vis", f"{reader.id_strs[i]}.png"), vis)

        # Optional display
        cv2.imshow("Prediction", vis[..., ::-1])
        cv2.waitKey(1)