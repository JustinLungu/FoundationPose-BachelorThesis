import sys
import os
import numpy as np
import trimesh
import imageio
import cv2

# Add both project root and FoundationPose to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FoundationPose')))


from FoundationPose.estimater import *
from FoundationPose.datareader import *

if __name__ == '__main__':
    ############## 1. SETUP OBJECT NAME AND PATHS ##############
    obj_name = "apple"
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_root = os.path.join(base_dir, "data", "HOTS_Processed_demo", obj_name)
    mesh_file = os.path.join(data_root, "mesh", "model.obj")
    test_scene_dir = data_root  # RGB, Depth, Mask, cam_K.txt are inside here

    debug_dir = os.path.join(base_dir, "results", "demo_run", obj_name)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "ob_in_cam"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "track_vis"), exist_ok=True)

    ############## 2. INITIALIZE MODEL & MESH ##############
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

    ############## 3. LOAD RGB-D FRAMES WITH INTRINSICS ##############
    reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

    ############## 4. POSE ESTIMATION LOOP ##############
    for i in range(len(reader.color_files)):
        print(f'Frame: {i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0:
            ############## FIRST FRAME: POSE REGISTRATION ##############
            mask = reader.get_mask(i).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
        else:
            ############## TRACKING ON SUBSEQUENT FRAMES ##############
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=2)

        ############## SAVE POSE ##############
        np.savetxt(os.path.join(debug_dir, "ob_in_cam", f"{reader.id_strs[i]}.txt"), pose.reshape(4, 4))

        ############## SAVE VISUALIZATION ##############
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        imageio.imwrite(os.path.join(debug_dir, "track_vis", f"{reader.id_strs[i]}.png"), vis)

        # Optional live view
        cv2.imshow("Prediction", vis[..., ::-1])
        cv2.waitKey(1)
