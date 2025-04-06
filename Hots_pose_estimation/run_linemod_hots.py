import os, sys

# === IMPORT PATH SETUP ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FoundationPose')))

# === FOUNDATIONPOSE IMPORTS ===
from FoundationPose.Utils import *
from FoundationPose.datareader import *
from FoundationPose.estimater import *

# === EXTERNAL LIBRARIES ===
import yaml
import cv2
import numpy as np
import trimesh
import logging
torch.cuda.set_device("cuda:0")

from utils.config import (
    DEBUG_LEVEL, DEVICE, OBJECT_ID,
    LINEMOD_DIR, MESH_DIR, DEBUG_DIR,
    DETECT_TYPE, USE_RECONSTRUCTED_MESH, REF_VIEW_DIR
)


# === FUNCTIONS ===
def get_mask(reader, i_frame, ob_id, detect_type):
    if detect_type == 'box':
        mask = reader.get_mask(i_frame, ob_id)
        H, W = mask.shape[:2]
        vs, us = np.where(mask > 0)
        umin, umax = us.min(), us.max()
        vmin, vmax = vs.min(), vs.max()
        valid = np.zeros((H, W), dtype=bool)
        valid[vmin:vmax, umin:umax] = 1

    elif detect_type == 'mask':
        mask = reader.get_mask(i_frame, ob_id)
        if mask is None:
            return None
        valid = mask > 0

    elif detect_type == 'detected':
        mask = cv2.imread(reader.color_files[i_frame].replace('rgb', 'mask_cosypose'), -1)
        valid = mask == ob_id

    else:
        raise RuntimeError

    return valid


def run_pose_estimation_worker(reader, i_frames, est: FoundationPose, debug=0, ob_id=None, device='cuda:0'):
    result = NestDict()
    est.to_device(device)
    est.glctx = dr.RasterizeCudaContext(device=device)

    for i, i_frame in enumerate(i_frames):
        logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")

        if ob_id != OBJECT_ID:
            continue

        video_id = reader.get_video_id()
        color = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)

        id_str = reader.id_strs[i_frame]
        frame_key = str(i_frame).zfill(6)
        if frame_key not in reader.K:
            logging.error(f"K matrix not found for frame {frame_key}. Skipping.")
            result[video_id][id_str][ob_id] = np.eye(4)
            continue

        K_matrix = np.array(reader.K[frame_key])
        ob_mask = get_mask(reader, i_frame, ob_id, detect_type=DETECT_TYPE)

        if ob_mask is None:
            logging.info("ob_mask not found, skip")
            result[video_id][id_str][ob_id] = np.eye(4)
            continue

        est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
        pose = est.register(K=K_matrix, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)

        if debug >= 3:
            m = est.mesh_ori.copy()
            tmp = m.copy()
            tmp.apply_transform(pose)
            tmp.export(f'{DEBUG_DIR}/model_tf.obj')

        result[video_id][id_str][ob_id] = pose

    return result


def run_pose_estimation():
    wp.force_load(device=DEVICE)
    debug = DEBUG_LEVEL
    res = NestDict()
    glctx = dr.RasterizeCudaContext()

    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
    est = FoundationPose(
        model_pts=mesh_tmp.vertices.copy(),
        model_normals=mesh_tmp.vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=mesh_tmp,
        scorer=None,
        refiner=None,
        glctx=glctx,
        debug_dir=DEBUG_DIR,
        debug=debug
    )

    reader_tmp = LinemodReader(LINEMOD_DIR, split=None)
    outs = []

    for ob_id in reader_tmp.ob_ids:
        ob_id = int(ob_id)
        if ob_id != OBJECT_ID:
            continue

        if USE_RECONSTRUCTED_MESH:
            mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=REF_VIEW_DIR)
        else:
            mesh = reader_tmp.get_gt_mesh(ob_id)

        symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]
        reader = LinemodReader(LINEMOD_DIR, split=None)
        est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
                         symmetry_tfs=symmetry_tfs, mesh=mesh)

        frame_batch = list(range(len(reader.color_files)))
        out = run_pose_estimation_worker(reader, frame_batch, est, debug, ob_id, DEVICE)
        outs.append(out)

    for out in outs:
        for video_id in out:
            for id_str in out[video_id]:
                for ob_id in out[video_id][id_str]:
                    res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

    with open(f'{DEBUG_DIR}/linemod_res.yml', 'w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)


# === MAIN ===
if __name__ == "__main__":
    os.makedirs(DEBUG_DIR, exist_ok=True)
    run_pose_estimation()