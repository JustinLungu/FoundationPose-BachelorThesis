# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This script is designed to run pose estimation using the FoundationPose model 
on the LINEMOD dataset, a popular dataset used for 6D object pose estimation. 
The script processes individual frames of RGB-D images, extracts the necessary 
information (like masks, camera intrinsics, and object poses), and then estimates 
the object pose for each frame using FoundationPose.
"""

from Utils import *
import os, sys
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
import yaml
import cv2
import numpy as np

# ======================== CONFIGURATION ========================
# List of Object IDs to process
# Available IDs: 1 = Gorilla, 4 = Camera, 6 = Cat, 8 = Drill, 9 = Duck, 10 = Eggbox
OBJECT_IDS = [4, 6, 9]  # Can specify multiple objects like [1, 6, 10]

class PathConfig:
    def __init__(self, code_dir, object_id):
        self.linemod_root = os.path.join(code_dir, 'Linemod_preprocessed')
        self.models_dir = os.path.join(self.linemod_root, 'models')
        self.data_dir = os.path.join(self.linemod_root, 'data')
        self.object_id = object_id
        
    @property
    def model_path(self):
        return os.path.join(self.models_dir, f'obj_{self.object_id}.ply')
        
    @property
    def data_path(self):
        return os.path.join(self.data_dir, str(self.object_id))
        
    @property
    def models_info_path(self):
        return os.path.join(self.models_dir, 'models_info.yml')

# ======================== HELPER FUNCTIONS ========================

def update_models_info_yml(ob_id, mesh, models_info_path):
    """Ensures the models_info.yml file exists and includes an entry for the given object ID."""
    from pathlib import Path

    ob_id = int(ob_id)
    bounding_box = mesh.bounding_box.bounds
    min_corner = bounding_box[0]
    max_corner = bounding_box[1]
    size = max_corner - min_corner
    diameter = np.linalg.norm(size)

    new_entry = {
        'diameter': float(diameter),
        'min_x': float(min_corner[0]), 'min_y': float(min_corner[1]), 'min_z': float(min_corner[2]),
        'size_x': float(size[0]), 'size_y': float(size[1]), 'size_z': float(size[2]),
    }

    print(f"New entry for object {ob_id}: {new_entry}")

    models_info_path = Path(models_info_path)
    if not models_info_path.exists():
        models_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(models_info_path, 'w') as f:
            f.write(f"{ob_id}: {new_entry}\n")
        print(f"[INFO] Created new models_info.yml with entry for object {ob_id}")
        return

    with open(models_info_path, 'r') as f:
        lines = f.readlines()
        models_info = {}
        for line in lines:
            if not line.strip():
                continue
            try:
                ob_id_str, entry_str = line.strip().split(": ", 1)
                ob_id = int(ob_id_str)
                entry = eval(entry_str)
                models_info[ob_id] = entry
            except ValueError:
                print(f"Warning: Skipping invalid line in models_info.yml: {line.strip()}")
                continue

    if ob_id not in models_info:
        models_info[ob_id] = new_entry
        with open(models_info_path, 'w') as f:
            for obj_id, entry in models_info.items():
                f.write(f"{obj_id}: {entry}\n")
        print(f"[INFO] Added object {ob_id} to models_info.yml")

def get_mask(reader, i_frame, ob_id, detect_type):
    """Extracts the object mask for a given frame and object ID."""
    if detect_type == 'box':
        mask = reader.get_mask(i_frame, ob_id)
        H, W = mask.shape[:2]
        vs, us = np.where(mask > 0)
        umin = us.min()
        umax = us.max()
        vmin = vs.min()
        vmax = vs.max()
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

# ======================== MAIN FUNCTIONALITY ========================

def run_pose_estimation_worker(reader, i_frames, est: FoundationPose = None, debug=0, ob_id=None, device='cuda:0'):
    """Runs pose estimation for a sequence of frames for a single object."""
    result = NestDict()
    torch.cuda.set_device(device)
    est.to_device(device)
    est.glctx = dr.RasterizeCudaContext(device=device)
    debug_dir = est.debug_dir

    for i, i_frame in enumerate(i_frames):
        logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")

        if ob_id not in OBJECT_IDS:  # Changed from single ID check to list check
            continue
        
        video_id = reader.get_video_id()
        color = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)

        if debug >= 5 and i_frame == 0:
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.imshow("Color Image2", color_bgr)
            depth_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype(np.uint8)
            cv2.imshow("Depth Image", depth_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        id_str = reader.id_strs[i_frame]
        H, W = color.shape[:2]
        frame_key = str(i_frame).zfill(6)
        
        if frame_key not in reader.K:
            logging.error(f"K matrix not found for frame {frame_key}. Skipping.")
            result[video_id][id_str][ob_id] = np.eye(4)
            continue
        
        K_matrix = np.array(reader.K[frame_key])
        ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
        
        if ob_mask is None:
            logging.info("ob_mask not found, skip")
            result[video_id][id_str][ob_id] = np.eye(4)
            continue
        
        est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
        pose = est.register(K=K_matrix, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
        
        if debug >= 2:
            logging.info(f"pose:\n{pose}")
        if debug >= 3:
            m = est.mesh_ori.copy()
            tmp = m.copy()
            tmp.apply_transform(pose)
            tmp.export(f'{debug_dir}/model_tf_{ob_id}.obj')  # Added object ID to filename

        result[video_id][id_str][ob_id] = pose

    return result

def run_pose_estimation():
    """Main function to run pose estimation pipeline."""
    wp.force_load(device='cuda')
    debug = opt.debug
    use_reconstructed_mesh = opt.use_reconstructed_mesh
    debug_dir = opt.debug_dir
    res = NestDict()
    glctx = dr.RasterizeCudaContext()

    # Temporary dummy box mesh for initializing FoundationPose
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
        debug=debug
    )

    # Process each object in the list
    for object_id in OBJECT_IDS:
        path_config = PathConfig(code_dir, object_id)

        # Handle models info file
        models_info_path = path_config.models_info_path
        os.makedirs(os.path.dirname(models_info_path), exist_ok=True)

        if not os.path.exists(models_info_path) or os.path.getsize(models_info_path) == 0:
            print(f"🔧 Generating initial models_info.yml for object {object_id}...")
            mesh = trimesh.load(path_config.model_path, force='mesh')
            update_models_info_yml(object_id, mesh, models_info_path)

        # Process object
        reader_tmp = LinemodReader(path_config.data_path, split=None)
        outs = []

        for ob_id in reader_tmp.ob_ids:
            ob_id = int(ob_id)
            if ob_id != object_id:  # Changed to check against current object_id
                continue

            if use_reconstructed_mesh:
                mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
            else:
                mesh = reader_tmp.get_gt_mesh(ob_id)

            update_models_info_yml(ob_id=ob_id, mesh=mesh, models_info_path=models_info_path)

            symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]
            reader = LinemodReader(path_config.data_path, split=None)
            video_id = reader.get_video_id()

            est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
                             symmetry_tfs=symmetry_tfs, mesh=mesh)

            frame_batch = list(range(len(reader.color_files)))
            out = run_pose_estimation_worker(reader, frame_batch, est, debug, ob_id, "cuda:0")
            outs.append(out)

        # Gather and save results for this object
        for out in outs:
            for video_id in out:
                for id_str in out[video_id]:
                    for ob_id in out[video_id][id_str]:
                        res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

    # Save all results
    with open(f'{opt.debug_dir}/linemod_res.yml', 'w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)

# ======================== MAIN ENTRY POINT ========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    print("CODE DIR", code_dir)

    # LINEMOD dataset configuration
    parser.add_argument('--linemod_dir', type=str, default="/Linemod_preprocessed", help="LINEMOD root directory")
    parser.add_argument('--use_reconstructed_mesh', type=int, default=0, help="Use reconstructed mesh or ground truth")
    parser.add_argument('--ref_view_dir', type=str, default="/Linemod_preprocessed/ref_views")
    parser.add_argument('--debug', type=int, default=1, help="Debug level")
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug', help="Directory to save debug info")

    opt = parser.parse_args()
    set_seed(0)
    detect_type = 'mask'  # Options: 'mask', 'box', 'detected'
    run_pose_estimation()