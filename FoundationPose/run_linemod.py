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
OBJECT_IDS = [1]  # Can specify multiple objects like [1, 6, 10]
MODELS_DIR = 'genAI_ply'  # Directory where the models are stored

class PathConfig:
    def __init__(self, code_dir, object_id):
        self.linemod_root = os.path.join(code_dir, 'Linemod_preprocessed')
        self.models_dir = os.path.join(self.linemod_root, MODELS_DIR)
        self.data_dir = os.path.join(self.linemod_root, 'data')
        self.object_id = object_id
        
    @property
    def model_path(self):
        return os.path.join(self.models_dir, f'obj_{self.object_id:02d}.ply')  # Added :02d formatting
        
    @property
    def data_path(self):
        return os.path.join(self.data_dir, f'{self.object_id:02d}')  # Added :02d formatting
        
    @property
    def models_info_path(self):
        return os.path.join(self.models_dir, 'models_info.yml')

# ======================== HELPER FUNCTIONS ========================

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

        # Process object
        try:
            reader_tmp = LinemodReader(path_config.data_path, split=None)
        except FileNotFoundError as e:
            print(f"Error loading data for object {object_id:02d}: {e}")
            print(f"Expected path: {path_config.data_path}")
            continue
            
        outs = []

        for ob_id in reader_tmp.ob_ids:
            ob_id = int(ob_id)
            if ob_id != object_id:
                continue

            if use_reconstructed_mesh:
                mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
            else:
                mesh = reader_tmp.get_gt_mesh(ob_id)

            symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]
            try:
                reader = LinemodReader(path_config.data_path, split=None)
            except FileNotFoundError as e:
                print(f"Error loading data for object {object_id:02d}: {e}")
                continue
                
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
    parser.add_argument('--debug', type=int, default=3, help="Debug level")
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/Linemod_results', help="Directory to save debug info")

    opt = parser.parse_args()
    set_seed(0)
    detect_type = 'mask'  # Options: 'mask', 'box', 'detected'
    run_pose_estimation()