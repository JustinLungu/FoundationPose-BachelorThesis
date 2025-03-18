# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml


class FoundationPose:
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/'):
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
    self.make_rotation_grid(min_n_views=40, inplane_step=60)

    self.glctx = glctx

    if scorer is not None:
      self.scorer = scorer
    else:
      self.scorer = ScorePredictor()

    if refiner is not None:
      self.refiner = refiner
    else:
      self.refiner = PoseRefinePredictor()

    self.pose_last = None   # Used for tracking; per the centered mesh


  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    """
    Resets the pose estimator for a new object. This involves adjusting the object's 
    mesh data, computing its diameter, voxelizing its point cloud, and preparing the 
    symmetry transformations (if any). This function prepares the 3D model for future 
    pose estimation operations by setting up various data structures.

    Args:
        model_pts (ndarray): Points representing the vertices of the object's 3D mesh.
        model_normals (ndarray): Normals for the vertices of the mesh.
        symmetry_tfs (ndarray, optional): Symmetry transformations for the object (if it has any). 
                                          Defaults to None.
        mesh (trimesh.Trimesh, optional): The 3D mesh of the object. If None, no mesh operations are done.

    Sets:
        - `self.mesh_ori`: A copy of the original mesh.
        - `self.diameter`: The computed diameter of the object based on the mesh.
        - `self.pts`: A voxelized version of the mesh's point cloud.
        - `self.normals`: The normalized vertex normals of the point cloud.
        - `self.symmetry_tfs`: Symmetry transformation matrices for the object.
    """

    # Compute the bounding box of the object's mesh.
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)

    # Compute the object's center.
    self.model_center = (min_xyz + max_xyz) / 2

    if mesh is not None:
        # Save a copy of the original mesh and center the mesh by subtracting its center from its vertices.
        self.mesh_ori = mesh.copy()
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)

    # Update the model points based on the modified (centered) mesh.
    model_pts = mesh.vertices

    # Compute the diameter of the object based on the mesh, used for scaling and binning.
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter / 20.0, 0.003)

    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')

    # Distance binning for matching depth or position data to the voxel grid.
    self.dist_bin = self.vox_size / 2

    # Angle bin size in degrees (used for rotation estimation).
    self.angle_bin = 20

    # Convert the model points to an Open3D point cloud and downsample it using the voxel size.
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)

    # Compute the bounding box of the downsampled point cloud.
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)

    # Convert the point cloud and normals into tensors for GPU processing.
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)

    logging.info(f'self.pts:{self.pts.shape}')

    # Save the mesh to a temporary file if it's not None.
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
        self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
        self.mesh.export(self.mesh_path)

    # Convert the mesh to GPU-compatible tensors (vertices, faces, etc.).
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    # Set up the symmetry transformation matrices, defaulting to identity if none are provided.
    if symmetry_tfs is None:
        self.symmetry_tfs = torch.eye(4).float().cuda()[None]  # Identity matrix for no symmetry
    else:
        self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")




  def get_tf_to_centered_mesh(self):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center


  def to_device(self, s='cuda:0'):
    """
    Transfers all relevant attributes of the FoundationPose object to the specified device (e.g., GPU or CPU).

    This function ensures that all tensors, models (e.g., neural networks), and mesh data within the object 
    are moved to the desired computational device for efficient processing. It also transfers any associated
    rendering context (glctx) to the device for rendering purposes.

    Parameters:
    -----------
    s : str
        The device to transfer the data to. Default is 'cuda:0', which refers to the first GPU.

    Operations:
    -----------
    1. Transfers all attributes of the object that are PyTorch tensors or models to the specified device.
    2. Transfers any mesh tensors stored in the `mesh_tensors` dictionary to the device.
    3. Transfers the `refiner` and `scorer` models (if present) to the device.
    4. Initializes the rendering context (glctx) on the specified device for rasterization.

    """
    # Loop through all attributes of the current object.
    for k in self.__dict__:
        # If the attribute is a PyTorch tensor or neural network module, move it to the specified device.
        if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
            #logging.info(f"Moving {k} to device {s}")
            self.__dict__[k] = self.__dict__[k].to(s)

    # Move any mesh-related tensors stored in the `mesh_tensors` dictionary to the specified device.
    for k in self.mesh_tensors:
        #logging.info(f"Moving {k} to device {s}")
        self.mesh_tensors[k] = self.mesh_tensors[k].to(s)

    # If the refiner model exists, move it to the specified device.
    if self.refiner is not None:
        self.refiner.model.to(s)

    # If the scorer model exists, move it to the specified device.
    if self.scorer is not None:
        self.scorer.model.to(s)

    # Move the rasterization context to the specified device if it's already initialized.
    if self.glctx is not None:
        self.glctx = dr.RasterizeCudaContext(s)




  def make_rotation_grid(self, min_n_views=40, inplane_step=60):
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)
    logging.info(f'cam_in_obs:{cam_in_obs.shape}')
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    logging.info(f"rot_grid:{rot_grid.shape}")
    rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
    logging.info(f"self.rot_grid: {self.rot_grid.shape}")


  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
    """
    Generates a set of random pose hypotheses for an object in the scene.

    This function creates initial guesses (hypotheses) for the object's pose by:
    - Using a predefined grid of rotation matrices (`self.rot_grid`).
    - Guessing the translation (position) of the object using the mask and depth map.
    - Assigning the guessed translation to the predefined rotation matrices to form a set of pose hypotheses.

    Returns:
    -------
    ob_in_cams : torch.Tensor
        A tensor representing the set of pose hypotheses for the object.
        Each pose is a 4x4 transformation matrix, where the top-left 3x3 submatrix is the rotation, 
        and the rightmost column is the translation vector.
    """

    # Clone the predefined grid of rotation matrices (self.rot_grid)
    ob_in_cams = self.rot_grid.clone()

    # Guess the translation (position) of the object using the mask and depth information.
    center = self.guess_translation(depth=depth, mask=mask, K=K)

    # Assign the guessed translation to the poses (the rightmost column of the 4x4 transformation matrix)
    # The translation vector is placed in the last column of the transformation matrices
    ob_in_cams[:, :3, 3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1, 3)

    # Return the set of pose hypotheses
    return ob_in_cams



  def guess_translation(self, depth, mask, K):
    """
    Guesses the translation vector (in 3D space) for an object by using the object's mask and depth information.

    This function estimates the translation (position) of the object in the scene by:
    - Identifying valid points in the mask and depth map.
    - Computing the center of the object in pixel coordinates.
    - Using the camera intrinsics (K matrix) to map the pixel coordinates and depth into 3D space.

    Parameters:
    ----------
    depth : np.ndarray
        The depth map for the current frame, where each pixel holds the depth value (distance to the camera) of the corresponding point in the scene.
    mask : np.ndarray
        A binary mask indicating where the object is located in the image (non-zero values mark the object).
    K : np.ndarray
        A 3x3 camera intrinsic matrix, which contains parameters like focal length and principal point of the camera.

    Returns:
    -------
    center : np.ndarray
        A 3D vector representing the estimated translation (position) of the object in the scene.
        If no valid points are found, it returns a zero vector (3,).
    """

    # Get the valid pixel coordinates where the mask is non-zero (i.e., where the object is present)
    vs, us = np.where(mask > 0)  # vs are the row indices (y-coordinates), us are the column indices (x-coordinates)
    
    # If the mask is empty (no valid pixels), return a zero vector
    if len(us) == 0:
        logging.info(f'mask is all zero')
        return np.zeros((3))
    
    # Compute the center of the mask in pixel coordinates (uc, vc are the pixel coordinates of the mask center)
    uc = (us.min() + us.max()) / 2.0
    vc = (vs.min() + vs.max()) / 2.0
    
    # Create a valid mask of pixels where both the mask is non-zero and the depth values are above a threshold (0.001)
    valid = mask.astype(bool) & (depth >= 0.001)
    
    # If no valid depth points are found, return a zero vector
    if not valid.any():
        logging.info(f"valid is empty")
        return np.zeros((3))

    # Compute the median depth value from the valid pixels (to avoid noisy depth values)
    zc = np.median(depth[valid])
    
    # Optionally print debugging information (uncomment the prints to enable)
    #print(f"[DEBUG] uc: {uc}, vc: {vc}, zc: {zc}")  # Prints the center pixel and depth
    #print(f"[DEBUG] K matrix: {K}, shape: {K.shape}")  # Prints the camera intrinsic matrix

    # Check if K is a valid 3x3 matrix. If not, raise an error.
    if K.ndim != 2 or K.shape != (3, 3):
        raise ValueError(f"Invalid K matrix. Expected 2D (3x3), got {K.shape}")
    
    # Use the inverse of the camera intrinsic matrix K to map the pixel coordinates (uc, vc) into 3D space
    # Multiply by zc (the median depth) to get the final 3D position (translation).
    center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc

    # Optionally save the 3D point cloud for debugging (if debug level is high)
    if self.debug >= 2:
        pcd = toOpen3dCloud(center.reshape(1, 3))  # Convert the center point into a point cloud
        o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)  # Save the point cloud to a file
    
    # Return the 3D center as a flattened array
    return center.reshape(3)



  def compare_matrices(self, aux, poses_np, tolerance=1e-6):
    """
    Compare two sets of 4x4 pose matrices (numpy and torch) element-wise.

    Parameters:
    aux (numpy.ndarray): The first set of matrices.
    poses (torch.Tensor): The second set of matrices (torch, which will be converted to numpy).
    tolerance (float): The acceptable tolerance for comparing floating point numbers.

    Returns:
    None: Prints differences if found or confirmation that the matrices are identical.
    """

    if aux.shape != poses_np.shape:
        print(f"Shape mismatch: aux shape = {aux.shape}, poses shape = {poses_np.shape}")
        return

    # Iterate through all matrices and compare each one
    for i in range(len(aux)):
        aux_matrix = aux[i]
        poses_matrix = poses_np[i]

        if not np.allclose(aux_matrix, poses_matrix, atol=tolerance):
            print(f"Difference detected at index {i}:")
            print(f"Matrix 1 (aux):\n{aux_matrix}")
            print(f"Matrix 2 (poses):\n{poses_matrix}")
            differences = aux_matrix - poses_matrix
            print(f"Differences:\n{differences}")
        else:
            print(f"Matrices at index {i} are identical within the tolerance.")


  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5):
    """
    Estimate the 6D pose of an object in a scene using RGB and depth data.
    
    This function performs the following steps:
    - Pre-processes the depth data with erosion and bilateral filtering.
    - Converts object masks from RGB to grayscale if necessary.
    - Estimates an initial guess for the object's translation.
    - Generates random pose hypotheses for the object.
    - Refines the pose using a mesh-based refiner model.
    - Scores the refined poses and selects the best one.

    Parameters:
    - K (numpy array): Camera intrinsic matrix (3x3).
    - rgb (numpy array): RGB image of the scene.
    - depth (numpy array): Depth image corresponding to the RGB image.
    - ob_mask (numpy array): Object mask, indicating the pixels that belong to the object.
    - ob_id (int, optional): Object ID (default is None).
    - glctx (optional): Rendering context for rasterization (default is None).
    - iteration (int, optional): Number of iterations for the refiner (default is 5).

    Returns:
    - best_pose (numpy array): The best estimated 4x4 pose matrix of the object in the scene.
    """
    set_seed(0)  # Set a random seed for reproducibility.
    logging.info('Welcome')

    # If a rendering context hasn't been created, create a new one using CUDA.
    if self.glctx is None:
        if glctx is None:
            self.glctx = dr.RasterizeCudaContext()  # Create CUDA rasterization context.
        else:
            self.glctx = glctx

    # Preprocess the depth image: apply erosion and bilateral filtering to smooth it.
    # this was only used for MIDAS depth images (don't forget to test with actual depth data)
    depth = depth[:, :, 0]  # Keep only the first channel
    print(f"Depth shape before erosion: {depth.shape}")
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    # If debugging is enabled, visualize the scene.
    if self.debug >= 2:
        #Converts the depth image into a 3D point cloud (XYZ map) based on the camera's intrinsic parameters.
        xyz_map = depth2xyzmap(depth, K)
        #Filters out invalid points and creates a point cloud of valid 3D points.
        valid = xyz_map[..., 2] >= 0.001
        pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])  # Convert valid points to a point cloud.
        #Saves the 3D point cloud to a .ply file for further analysis or visualization.
        o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply', pcd)
        #Saves the object mask as a grayscale image
        cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask * 255.0).clip(0, 255))

    # Convert object mask from RGB to grayscale if necessary.
    if ob_mask.ndim == 3:
        ob_mask = ob_mask[:, :, 0]  # Convert to grayscale by taking the first channel.

    normal_map = None  # Initialize the normal map (optional, not used here).

    # Create a mask for valid points (where the object mask is non-zero and depth is positive).
    valid = (depth >= 0.001) & (ob_mask > 0)
    
    # If there are too few valid points, return an identity pose with a guessed translation.
    if valid.sum() < 4:
        logging.info('Valid points too small, return')
        pose = np.eye(4)  # Identity matrix for the pose.
        pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)  # Guess translation from depth.
        return pose

    # Additional visualization for debugging.
    if self.debug >= 2:
        imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
        cv2.imwrite(f'{self.debug_dir}/depth.png', (depth * 1000).astype(np.uint16))
        valid = xyz_map[..., 2] >= 0.001
        pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
        o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply', pcd)

    # Set the image dimensions and camera intrinsics.
    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    # Generate random pose hypotheses for the object.
    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    poses = poses.data.cpu().numpy()  # Convert pose hypotheses to numpy.

    # Compute the initial pose error with respect to the ground truth.
    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"After viewpoint generation, add_errs min: {add_errs.min()}")
    
    #Convert a depth map to a 3D point cloud (XYZ map) in camera space
    xyz_map = depth2xyzmap(depth, K)

    
    # Refine the pose using the refiner model.
    poses, vis = self.refiner.predict(
        mesh=self.mesh,
        mesh_tensors=self.mesh_tensors,
        rgb=rgb,
        depth=depth,
        K=K,
        ob_in_cams=poses, #.data.cpu().numpy(),
        normal_map=normal_map,
        xyz_map=xyz_map,
        glctx=self.glctx,
        mesh_diameter=self.diameter,
        iteration=iteration,
        get_vis=self.debug >= 2
    )

    # If visualization is returned by the refiner, save it.
    if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)
    
    # Score the refined poses using the scorer model.
    scores, vis = self.scorer.predict(
        mesh=self.mesh,
        rgb=rgb,
        depth=depth,
        K=K,
        ob_in_cams=poses, #.data.cpu().numpy(),
        normal_map=normal_map,
        mesh_tensors=self.mesh_tensors,
        glctx=self.glctx,
        mesh_diameter=self.diameter,
        get_vis=self.debug >= 2
    )
    
    # If visualization is returned by the scorer, save it.
    if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)
    
    # Compute the final pose error after refinement.
    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"Final pose estimation, add_errs min: {add_errs.min()}")

    # Sort the poses based on their scores.
    #ids = torch.as_tensor(scores).cpu().argsort(descending=True)
    #scores = scores[ids].cpu()
    ids = torch.as_tensor(scores).argsort(descending=True)
    scores = scores[ids]
    poses = poses[ids]
    if self.debug >= 2:
        logging.info(f'Sorted pose IDs: {ids}')
        logging.info(f'Sorted scores: {scores}')
    
    # Select the best pose (the highest-scoring one).
    #best_pose = poses[0] @ self.get_tf_to_centered_mesh().cpu().numpy()
    best_pose = poses[0] @ self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.best_id = ids[0]
    self.poses = poses
    self.scores = scores

    # Return the best pose as a numpy array.
    #return best_pose
    return best_pose.data.cpu().numpy()


  def compute_add_err_to_gt_pose(self, poses):
    '''
    @poses: wrt. the centered mesh
    '''
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)


  def track_one(self, rgb, depth, K, iteration, extra={}):
    """
    Track the object in the scene based on the previous pose (`pose_last`) and update the pose for the current frame.
    
    This function refines the previously estimated pose using the RGB and depth information from the current frame.
    The `refiner` model is used to predict the pose for the current frame based on the last known pose.
    
    Parameters:
    - rgb (numpy array): RGB image of the scene.
    - depth (numpy array): Depth image corresponding to the RGB image.
    - K (numpy array): Camera intrinsic matrix (3x3).
    - iteration (int): Number of iterations for pose refinement.
    - extra (dict, optional): Extra outputs for debugging or visualization.
    
    Returns:
    - pose (numpy array): The updated 4x4 pose matrix of the object in the scene.
    
    Raises:
    - RuntimeError: If `pose_last` is not initialized (i.e., no previous pose has been estimated).
    """
    # If the last pose is not available, raise an error.
    if self.pose_last is None:
        logging.info("Please init pose by register first")
        raise RuntimeError

    logging.info("Welcome")

    # Convert the depth map to a PyTorch tensor and move it to the GPU (cuda).
    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)

    # Preprocess the depth image: apply erosion and bilateral filtering to smooth it.
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    logging.info("Depth processing done")

    # Convert the depth map to an XYZ map (3D coordinates for each pixel) using camera intrinsics.
    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    # Use the `refiner` model to refine the pose for the current frame based on the last known pose (`pose_last`).
    # The pose is predicted using the mesh, RGB image, depth map, XYZ map, and camera intrinsics.
    pose, vis = self.refiner.predict(
        mesh=self.mesh,
        mesh_tensors=self.mesh_tensors,
        rgb=rgb,
        depth=depth,
        K=K,
        ob_in_cams=self.pose_last.reshape(1, 4, 4).data.cpu().numpy(),  # Last known pose
        normal_map=None,
        xyz_map=xyz_map,
        mesh_diameter=self.diameter,
        glctx=self.glctx,
        iteration=iteration,
        get_vis=self.debug >= 2
    )

    logging.info("Pose prediction done")

    # If debugging is enabled, store the visualization results in the `extra` dictionary.
    if self.debug >= 2:
        extra['vis'] = vis

    # Update `pose_last` with the new pose for the current frame.
    self.pose_last = pose

    # Return the updated pose after applying the transformation to the centered mesh.
    return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)

