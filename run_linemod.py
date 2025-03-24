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


Key Requirements for FoundationPose Model to Predict:
To run FoundationPose for pose estimation, the following is required:

  - Object Mesh (3D model): Either the ground-truth mesh or a reconstructed mesh.
  - Camera Intrinsics (K Matrix): These must be extracted for each frame.
  - Object Mask: A binary mask that highlights the region where the object is present.
  - RGB and Depth Images: For each frame, the pose estimation uses both RGB and depth data.
  - Ground Truth Pose (optional): Only needed if you want to compare or evaluate the accuracy of predictions.
"""



"""
datareader.py and estimater.py handle reading the data and estimating the pose using FoundationPose.
Utils contains helper functions related to the model or data processing.
"""
from Utils import *
import os,sys
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml

import cv2
import numpy as np

OBJECT_ID = 1

def update_models_info_yml(ob_id, mesh, models_info_path):
    """
    Ensures the models_info.yml file exists and includes an entry for the given object ID.
    If not, creates or appends the relevant entry using the mesh.

    Parameters:
    - ob_id: The object ID (as int)
    - mesh: The trimesh object
    - models_info_path: Path to the models_info.yml file
    """
    from pathlib import Path

    ob_id = int(ob_id)

    # Compute bounding box and diameter
    bounding_box = mesh.bounding_box.bounds
    min_corner = bounding_box[0]
    max_corner = bounding_box[1]
    size = max_corner - min_corner
    diameter = np.linalg.norm(size)

    # Build new entry
    new_entry = {
        'diameter': float(diameter),
        'min_x': float(min_corner[0]), 'min_y': float(min_corner[1]), 'min_z': float(min_corner[2]),
        'size_x': float(size[0]), 'size_y': float(size[1]), 'size_z': float(size[2]),
    }

    # Debugging: Print the new entry
    print(f"New entry for object {ob_id}: {new_entry}")

    # Create file if it doesn't exist
    models_info_path = Path(models_info_path)
    if not models_info_path.exists():
        models_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(models_info_path, 'w') as f:
            # Write the entry in the desired format (single line)
            f.write(f"{ob_id}: {new_entry}\n")
        print(f"[INFO] Created new models_info.yml with entry for object {ob_id}")
        return

    # Load existing YAML
    with open(models_info_path, 'r') as f:
        lines = f.readlines()
        models_info = {}
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            # Parse each line manually
            try:
                ob_id_str, entry_str = line.strip().split(": ", 1)
                ob_id = int(ob_id_str)
                entry = eval(entry_str)  # Convert the string representation of the dictionary to a dictionary
                models_info[ob_id] = entry
            except ValueError:
                print(f"Warning: Skipping invalid line in models_info.yml: {line.strip()}")
                continue

    # If entry is missing, add it
    if ob_id not in models_info:
        models_info[ob_id] = new_entry
        with open(models_info_path, 'w') as f:
            # Write all entries in the desired format (single line)
            for obj_id, entry in models_info.items():
                f.write(f"{obj_id}: {entry}\n")
        print(f"[INFO] Added object {ob_id} to models_info.yml")


def get_mask(reader, i_frame, ob_id, detect_type):
  """
  This function extracts the object mask for a given frame and object ID. 
  Depending on the detection type (box, mask, or detected), it processes 
  the mask differently:

  box: Extracts a bounding box around the object.
  mask: Uses the binary mask for the object.
  detected: Loads a pre-generated mask file from disk.

  The mask will be used later in the pipeline to constrain where the model looks for the object.
  """

  # Case 1: If the detection type is 'box', we are manually constructing a bounding box
  if detect_type == 'box':
    # Get the object mask for the given frame and object ID from the reader (could be a binary mask)
    mask = reader.get_mask(i_frame, ob_id)
    H, W = mask.shape[:2]  # Get the height and width of the mask
    vs, us = np.where(mask > 0)  # Find the pixels where the mask is non-zero (object area)
    
    #bounding box coordinates around the object
    #us = x-coordinate, vs = y-coordinate
    umin = us.min()  
    umax = us.max()  
    vmin = vs.min()  
    vmax = vs.max()  
    
    # Create a valid mask of zeros (same size as the image) and set the object area to 1
    valid = np.zeros((H, W), dtype=bool)  # init empty boolean mask (all False)
    valid[vmin:vmax, umin:umax] = 1  #set region inside bounding box to True

  # Case 2: If the detection type is 'mask', we are using a pre-existing binary mask
  elif detect_type == 'mask':
    # Get the object mask for the given frame and object ID from the reader
    mask = reader.get_mask(i_frame, ob_id)
    if mask is None:
      return None  # If no mask is found, return None to indicate that the object wasn't detected
    
    # Convert the mask into a boolean array where pixels with value > 0 are considered valid
    valid = mask > 0

  # Case 3: If the detection type is 'detected', load a pre-generated mask file from disk
  elif detect_type == 'detected':
    # Load the mask file from the disk (using the color file path but replacing 'rgb' with 'mask_cosypose')
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb', 'mask_cosypose'), -1)
    
    # Check if the mask value matches the object ID (creating a boolean mask)
    valid = mask == ob_id

  #invalid detection type --> raise an error
  else:
    raise RuntimeError

  return valid  # valid mask: a boolean array indicating where the object is


def run_pose_estimation_worker(reader, i_frames, est: FoundationPose = None, debug=0, ob_id=None, device='cuda:0'):
    """
    This function runs pose estimation for a sequence of frames for a single object. 

    It performs the following steps:
    - Sets up the environment for pose estimation, such as configuring the GPU device and initializing rendering context.
    - Iterates over the frames, fetching image data, camera intrinsics (K matrix), and object masks for each frame.
    - Uses the FoundationPose model to estimate the 6D pose of the object in the frame.
    - Optionally outputs a 3D visualization of the objects predicted pose for debugging purposes.
    - Stores the pose result in a nested dictionary for each frame and object.

    Parameters:
    - reader: Data reader object, providing access to images, masks, intrinsics, etc.
    - i_frames: List of frame indices to process.
    - est: Instance of FoundationPose model, used to perform the actual pose estimation.
    - debug: Debug level, controls the amount of logging and visualization.
    - ob_id: Object ID to process.
    - device: Device to run the computations on (default is 'cuda:0').

    Returns:
    - result: Nested dictionary containing pose estimates for each frame and object.
    """
    # Initialize the result storage, a nested dictionary to store pose estimates per frame
    result = NestDict()

    # Set the GPU device where computations will be executed
    torch.cuda.set_device(device)

    # Send the pose estimation model to the GPU
    est.to_device(device)

    
    # Initialize a rendering context for rasterization (for rendering the object during estimation)
    # import nvdiffrast.torch as dr coming from Utils
    est.glctx = dr.RasterizeCudaContext(device=device)

    # Store the directory for debugging (where files may be saved)
    debug_dir = est.debug_dir

    # Loop over each frame index in the i_frames list
    for i, i_frame in enumerate(i_frames):
      logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")

      #NOTE: remove this when you want more than just one object of the dataset
      #or change the number if you want a different object
      # Limit processing to object ID 1 (or another desired object ID)
      if ob_id != OBJECT_ID:
          continue  # Skip other objects if it's not the desired object
      
      # Get the video ID, color image, and depth image for the current frame
      video_id = reader.get_video_id()
      color = reader.get_color(i_frame)
      #pixel = 0 means no valid depth information
      #pixel = 0.638 means that the pixels are at depth 0.638 meters
      depth = reader.get_depth(i_frame)

      if debug >= 5 and i_frame == 0:
        #show coloured image (convert RGB to BGR)
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imshow("Color Image2", color_bgr)

        #show the depth image
        # Normalize the depth image to fit the 0-255 range for display
        depth_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = depth_display.astype(np.uint8)
        cv2.imshow("Depth Image", depth_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

      # Get the string ID for the current frame (might be frame number as a string)
      id_str = reader.id_strs[i_frame]
      
      # Get the height and width of the color image
      H, W = color.shape[:2]

      # Extract the camera intrinsic matrix (K matrix) for the current frame as a NumPy array
      frame_key = str(i_frame).zfill(6)  # Zero-pad the frame number to match dictionary keys
      if frame_key not in reader.K:
          logging.error(f"K matrix not found for frame {frame_key}. Skipping.")
          result[video_id][id_str][ob_id] = np.eye(4)  # Return an identity matrix if K matrix is not found
          continue
      
      # Convert the K matrix to a NumPy array
      K_matrix = np.array(reader.K[frame_key])

      # Get the object mask for the current frame and object ID using the `get_mask` function
      ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
      if ob_mask is None:
          logging.info("ob_mask not found, skip")
          result[video_id][id_str][ob_id] = np.eye(4)  # Return an identity matrix if the mask is not found
          continue
      
      # Retrieve the ground truth pose for the object in the current frame (if available)
      est.gt_pose = reader.get_gt_pose(i_frame, ob_id)

      
      # Perform pose estimation using the FoundationPose model's `register` function
      # register = "do inference"
      pose = est.register(K=K_matrix, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
      if debug >= 2:
        logging.info(f"pose:\n{pose}")

      # If debugging level is high (>= 3), save a transformed version of the object mesh
      if debug >= 3:
          m = est.mesh_ori.copy()  # Make a copy of the original mesh
          tmp = m.copy()
          tmp.apply_transform(pose)  # Apply the estimated transformation to the mesh
          tmp.export(f'{debug_dir}/model_tf.obj')  # Export the transformed mesh for visualization

      # Store the estimated pose in the result dictionary for this frame and object
      result[video_id][id_str][ob_id] = pose

      #NOTE: remove this break once you want to do video-like demo
      #for now we only run FoundationPose on only one frame of one object
      #break

    # Return the result dictionary, which contains the pose estimates for each frame and object
    return result


def run_pose_estimation():
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

    models_info_path = os.path.join(code_dir, 'demo_linemod/models/models_info.yml')
    os.makedirs(os.path.dirname(models_info_path), exist_ok=True)

    # ✅ Pre-populate models_info.yml if missing or empty
    if not os.path.exists(models_info_path) or os.path.getsize(models_info_path) == 0:
        print("🔧 Generating initial models_info.yml before reader loads it...")
        mesh_path = os.path.join(code_dir, 'demo_linemod/models/apple_fixed.obj')
        mesh = trimesh.load(mesh_path, force='mesh')
        update_models_info_yml(OBJECT_ID, mesh, models_info_path)

    # ✅ Now safe to load reader
    reader_tmp = LinemodReader(f'demo_linemod/data/01', split=None)
    outs = []

    for ob_id in reader_tmp.ob_ids:
        ob_id = int(ob_id)
        if ob_id != OBJECT_ID:
            continue

        # Load mesh
        if use_reconstructed_mesh:
            mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
        else:
            mesh = reader_tmp.get_gt_mesh(ob_id)

        # ✅ Ensure models_info is up to date (won't overwrite previous logic)
        update_models_info_yml(ob_id=ob_id, mesh=mesh, models_info_path=models_info_path)

        symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]
        video_dir = 'demo_linemod/data/01'
        reader = LinemodReader(video_dir, split=None)
        video_id = reader.get_video_id()

        est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
                         symmetry_tfs=symmetry_tfs, mesh=mesh)

        frame_batch = list(range(len(reader.color_files)))
        out = run_pose_estimation_worker(reader, frame_batch, est, debug, ob_id, "cuda:0")
        outs.append(out)

    # Gather and save results
    for out in outs:
        for video_id in out:
            for id_str in out[video_id]:
                for ob_id in out[video_id][id_str]:
                    res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

    with open(f'{opt.debug_dir}/linemod_res.yml', 'w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)



if __name__ == '__main__':
    """
    This is the entry point of the script. It sets up the command-line arguments, configures settings 
    like directories and debug options, and calls the `run_pose_estimation` function to start the 
    6D pose estimation process on the LINEMOD dataset.
    """
    
    # Create an argument parser to allow the user to provide configuration options from the command line.
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    print("CODE DIR", code_dir)

    # Define command-line arguments that can be passed to the script:
    
    # ================================= linemod dataset =================================
    # parser.add_argument('--linemod_dir', type=str, default="/Linemod_preprocessed", help="LINEMOD root directory")
    # #choose whether to use reconstructed meshes (1) or the ground truth meshes (0, default).
    # parser.add_argument('--use_reconstructed_mesh', type=int, default=0, help="Use reconstructed mesh or ground truth")
    # # directory containing reference views for mesh reconstruction (default path provided).
    # parser.add_argument('--ref_view_dir', type=str, default="/Linemod_preprocessed/ref_views")
    # parser.add_argument('--debug', type=int, default=5, help="Debug level")
    # parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug', help="Directory to save debug info")

    # ================================= custom object dataset =================================
    parser.add_argument('--linemod_dir', type=str, default=f'{code_dir}/demo_linemod/data/01', help="Custom object root directory")
    # Choose whether to use reconstructed meshes (1) or the ground truth meshes (0, default)
    parser.add_argument('--use_reconstructed_mesh', type=int, default=0, help="Use reconstructed mesh or ground truth")
    # This can be ignored or pointed to a dummy path if not using reconstruction
    parser.add_argument('--ref_view_dir', type=str, default=f'{code_dir}/demo_linemod/ref_views', help="Directory with reference views")
    # Debug options
    parser.add_argument('--debug', type=int, default=5, help="Debug level")
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/demo_linemod/data/01/debug', help="Directory to save debug info")



    opt = parser.parse_args()
    set_seed(0)

    # Define the type of detection to be used in the pose estimation process. 
    # This determines how the object will be detected in the images.
    # Options include:
    # - 'mask': Uses a pre-computed binary mask for each object.
    # - 'box': Uses a bounding box around the object.
    # - 'detected': Uses a pre-generated mask from another detector (e.g., CosyPose).
    detect_type = 'mask'
    run_pose_estimation()
