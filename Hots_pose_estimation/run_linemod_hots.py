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
    DEBUG_LEVEL, DEVICE, 
    LINEMOD_DIR, MESH_DIR, DEBUG_DIR,
    DETECT_TYPE, USE_RECONSTRUCTED_MESH, REF_VIEW_DIR,
    PROCESS_ALL_OBJECTS, CUSTOM_OBJECT_IDS
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
        try:
            logging.info(f"Processing frame {i+1}/{len(i_frames)} (Frame {i_frame}) for object {ob_id}")

            video_id = reader.get_video_id()
            color = reader.get_color(i_frame)
            depth = reader.get_depth(i_frame)

            id_str = reader.id_strs[i_frame]
            frame_key = str(i_frame).zfill(6)
            
            # Skip if K matrix not available
            if frame_key not in reader.K:
                logging.warning(f"K matrix not found for frame {frame_key}. Skipping.")
                result[video_id][id_str][ob_id] = np.eye(4)
                continue

            K_matrix = np.array(reader.K[frame_key])
            ob_mask = get_mask(reader, i_frame, ob_id, detect_type=DETECT_TYPE)

            # Skip if mask not available
            if ob_mask is None:
                logging.warning(f"Mask not found for object {ob_id} in frame {i_frame}. Skipping.")
                result[video_id][id_str][ob_id] = np.eye(4)
                continue

            # Store GT pose for debugging if available
            try:
                est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
            except:
                est.gt_pose = None

            # Run pose estimation
            pose = est.register(
                K=K_matrix,
                rgb=color,
                depth=depth,
                ob_mask=ob_mask,
                ob_id=ob_id
            )

            # Debug output if enabled
            if debug >= 3:
                debug_output_dir = os.path.join(DEBUG_DIR, f"object_{ob_id}")
                os.makedirs(debug_output_dir, exist_ok=True)
                
                m = est.mesh_ori.copy()
                tmp = m.copy()
                tmp.apply_transform(pose)
                tmp.export(f'{debug_output_dir}/frame_{i_frame}_model_tf.obj')
                
                # Save visualization image
                vis_img = est.last_vis
                if vis_img is not None:
                    cv2.imwrite(f'{debug_output_dir}/frame_{i_frame}_vis.png', vis_img)

            result[video_id][id_str][ob_id] = pose

        except Exception as e:
            logging.error(f"Error processing frame {i_frame} for object {ob_id}: {str(e)}")
            # Store identity matrix as fallback
            if video_id and id_str:
                result[video_id][id_str][ob_id] = np.eye(4)
            continue

    return result


def run_pose_estimation():
    wp.force_load(device=DEVICE)
    debug = DEBUG_LEVEL
    res = NestDict()
    glctx = dr.RasterizeCudaContext()

    # Create temporary mesh for initialization
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

    # Get list of object directories
    object_dirs = [d for d in os.listdir(LINEMOD_DIR) 
                if os.path.isdir(os.path.join(LINEMOD_DIR, d)) and d.isdigit()]
    
    # Determine which objects to process based on config
    if PROCESS_ALL_OBJECTS:
        objects_to_process = object_dirs
    else:
        objects_to_process = [f"{ob_id:02d}" for ob_id in CUSTOM_OBJECT_IDS]  # Format as 2-digit strings

    logging.info(f"Processing objects: {objects_to_process}")
    outs = []

    for obj_dir in objects_to_process:
        try:
            ob_id = int(obj_dir)  # Convert directory name to integer ID
            logging.info(f"Processing object ID: {ob_id} from directory {obj_dir}")
            
            # Initialize reader for this object's directory
            obj_path = os.path.join(LINEMOD_DIR, obj_dir)
            reader = LinemodReader(obj_path, split=None)
            
            # Load appropriate mesh
            if USE_RECONSTRUCTED_MESH:
                mesh = reader.get_reconstructed_mesh(ob_id, ref_view_dir=REF_VIEW_DIR)
            else:
                mesh = reader.get_gt_mesh(ob_id)

            if mesh is None:
                logging.warning(f"Mesh not found for object {ob_id}, skipping")
                continue

            # Get symmetry transforms (if any)
            symmetry_tfs = reader.symmetry_tfs.get(ob_id, None)
            
            # Configure the estimator for this object
            est.reset_object(
                model_pts=mesh.vertices.copy(),
                model_normals=mesh.vertex_normals.copy(),
                symmetry_tfs=symmetry_tfs,
                mesh=mesh
            )

            # Process all frames for this object
            frame_batch = list(range(len(reader.color_files)))
            out = run_pose_estimation_worker(reader, frame_batch, est, debug, ob_id, DEVICE)
            outs.append(out)

        except Exception as e:
            logging.error(f"Error processing object {obj_dir}: {str(e)}")
            continue

    # Combine all results
    for out in outs:
        for video_id in out:
            for id_str in out[video_id]:
                for ob_id in out[video_id][id_str]:
                    res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

    # Save results to YAML file
    result_file = f'{DEBUG_DIR}/linemod_res.yml'
    with open(result_file, 'w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)
    
    logging.info(f"Pose estimation completed. Results saved to {result_file}")

# === MAIN ===
if __name__ == "__main__":
    os.makedirs(DEBUG_DIR, exist_ok=True)
    run_pose_estimation()