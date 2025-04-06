import os
import torch
import logging
import numpy as np
import trimesh
import yaml
import cv2
import sys

# Add both project root and FoundationPose to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FoundationPose')))

from FoundationPose.datareader import LinemodReader
from FoundationPose.estimater import FoundationPose
from FoundationPose.Utils import NestDict, make_yaml_dumpable
import nvdiffrast.torch as dr

# Hardcoded paths – adjust as needed
code_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(code_dir, "data/HOTS_Processed_linemod/data/01")
mesh_dir = os.path.join(code_dir, "data/HOTS_Processed_linemod/models")
debug_dir = os.path.join(code_dir, "results/linemod_run/apple")

# FoundationPose options
use_reconstructed_mesh = False
debug_level = 5
detect_type = 'mask'
device = 'cuda:0'

os.makedirs(debug_dir, exist_ok=True)

def get_mask(reader, i_frame, ob_id, detect_type):
    if detect_type == 'mask':
        mask = reader.get_mask(i_frame, ob_id)
        return mask > 0 if mask is not None else None
    else:
        raise NotImplementedError("Only 'mask' is supported in this script.")

def run_pose_estimation():
    torch.cuda.set_device(device)
    glctx = dr.RasterizeCudaContext(device=device)
    res = NestDict()

    # Init dummy mesh for estimator
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
    est = FoundationPose(
        model_pts=mesh_tmp.vertices.copy(),
        model_normals=mesh_tmp.vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=mesh_tmp,
        scorer=None,
        refiner=None,
        glctx=glctx,
        debug_dir=debug_dir,
        debug=debug_level
    )

    # Read one sequence (object ID = 1 for apple)
    ob_id = 1
    reader = LinemodReader(dataset_dir, split=None)
    mesh_path = os.path.join(mesh_dir, "obj_{:02d}.ply".format(ob_id))
    mesh = trimesh.load(mesh_path, force='mesh')
    symmetry_tfs = reader.symmetry_tfs.get(ob_id, None)

    est.reset_object(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        symmetry_tfs=symmetry_tfs,
        mesh=mesh
    )

    # Main loop over frames
    for i_frame in range(len(reader.color_files)):
        id_str = reader.id_strs[i_frame]
        video_id = reader.get_video_id()

        rgb = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)
        frame_key = int(reader.id_strs[i_frame].lstrip("0") or "0")
        K = reader.K_table.get(frame_key, reader.K)

        mask = get_mask(reader, i_frame, ob_id, detect_type)

        if mask is None:
            print(f"[SKIP] No mask found for frame {i_frame}")
            res[video_id][id_str][ob_id] = np.eye(4)
            continue

        est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
        pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, ob_id=ob_id)

        res[video_id][id_str][ob_id] = pose

    # Save results
    result_path = os.path.join(debug_dir, "linemod_res.yml")
    with open(result_path, 'w') as f:
        yaml.safe_dump(make_yaml_dumpable(res), f)

if __name__ == "__main__":
    run_pose_estimation()
