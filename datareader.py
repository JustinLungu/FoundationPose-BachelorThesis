# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import json,os,sys
import yaml


BOP_LIST = ['lmo','tless','ycbv','hb','tudl','icbin','itodd']
BOP_DIR = os.getenv('BOP_DIR')

def get_bop_reader(video_dir, zfar=np.inf):
  if 'ycbv' in video_dir or 'YCB' in video_dir:
    return YcbVideoReader(video_dir, zfar=zfar)
  if 'lmo' in video_dir or 'LINEMOD-O' in video_dir:
    return LinemodOcclusionReader(video_dir, zfar=zfar)
  if 'tless' in video_dir or 'TLESS' in video_dir:
    return TlessReader(video_dir, zfar=zfar)
  if 'hb' in video_dir:
    return HomebrewedReader(video_dir, zfar=zfar)
  if 'tudl' in video_dir:
    return TudlReader(video_dir, zfar=zfar)
  if 'icbin' in video_dir:
    return IcbinReader(video_dir, zfar=zfar)
  if 'itodd' in video_dir:
    return ItoddReader(video_dir, zfar=zfar)
  else:
    raise RuntimeError


def get_bop_video_dirs(dataset):
  if dataset=='ycbv':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/ycbv/test/*'))
  elif dataset=='lmo':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/lmo/lmo_test_bop19/test/*'))
  elif dataset=='tless':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/tless/tless_test_primesense_bop19/test_primesense/*'))
  elif dataset=='hb':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/hb/hb_test_primesense_bop19/test_primesense/*'))
  elif dataset=='tudl':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/tudl/tudl_test_bop19/test/*'))
  elif dataset=='icbin':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/icbin/icbin_test_bop19/test/*'))
  elif dataset=='itodd':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/itodd/itodd_test_bop19/test/*'))
  else:
    raise RuntimeError
  return video_dirs



class YcbineoatReader:
  def __init__(self,video_dir, downscale=1, shorter_side=None, zfar=np.inf):
    self.video_dir = video_dir
    self.downscale = downscale
    self.zfar = zfar
    self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
    self.K = np.loadtxt(f'{video_dir}/cam_K.txt').reshape(3,3)
    self.id_strs = []
    for color_file in self.color_files:
      id_str = os.path.basename(color_file).replace('.png','')
      self.id_strs.append(id_str)
    self.H,self.W = cv2.imread(self.color_files[0]).shape[:2]

    if shorter_side is not None:
      self.downscale = shorter_side/min(self.H, self.W)

    self.H = int(self.H*self.downscale)
    self.W = int(self.W*self.downscale)
    self.K[:2] *= self.downscale

    self.gt_pose_files = sorted(glob.glob(f'{self.video_dir}/annotated_poses/*'))

    self.videoname_to_object = {
      'bleach0': "021_bleach_cleanser",
      'bleach_hard_00_03_chaitanya': "021_bleach_cleanser",
      'cracker_box_reorient': '003_cracker_box',
      'cracker_box_yalehand0': '003_cracker_box',
      'mustard0': '006_mustard_bottle',
      'mustard_easy_00_02': '006_mustard_bottle',
      'sugar_box1': '004_sugar_box',
      'sugar_box_yalehand0': '004_sugar_box',
      'tomato_soup_can_yalehand0': '005_tomato_soup_can',
    }


  def get_video_name(self):
    return self.video_dir.split('/')[-1]

  def __len__(self):
    return len(self.color_files)

  def get_gt_pose(self,i):
    try:
      pose = np.loadtxt(self.gt_pose_files[i]).reshape(4,4)
      return pose
    except:
      logging.info("GT pose not found, return None")
      return None


  def get_color(self,i):
    """
    Reads and processes the color image at the specified index.

    This function performs the following steps:
    - Reads the color image from the file at the given index 'i' in the 'self.color_files' list.
    - The image is read using the 'imageio' library, and the last channel is removed to ensure it returns a 3-channel RGB image.
    - Resizes the image to the target width (self.W) and height (self.H) using OpenCV's nearest-neighbor interpolation.

    Args:
        i (int): The index of the color image in the 'self.color_files' list to read.

    Returns:
        numpy.ndarray: The processed RGB color image.
    """
    color = imageio.imread(self.color_files[i])[...,:3]
    color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return color

  def get_mask(self,i):
    mask = cv2.imread(self.color_files[i].replace('rgb','masks'),-1)
    if len(mask.shape)==3:
      for c in range(3):
        if mask[...,c].sum()>0:
          mask = mask[...,c]
          break
    mask = cv2.resize(mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
    return mask

  def get_depth(self,i):
    depth = cv2.imread(self.color_files[i].replace('rgb','depth'),-1)/1e3
    depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    depth[(depth<0.001) | (depth>=self.zfar)] = 0
    return depth


  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.K)
    return xyz_map

  def get_occ_mask(self,i):
    hand_mask_file = self.color_files[i].replace('rgb','masks_hand')
    occ_mask = np.zeros((self.H,self.W), dtype=bool)
    if os.path.exists(hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(hand_mask_file,-1)>0)

    right_hand_mask_file = self.color_files[i].replace('rgb','masks_hand_right')
    if os.path.exists(right_hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(right_hand_mask_file,-1)>0)

    occ_mask = cv2.resize(occ_mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST)

    return occ_mask.astype(np.uint8)

  def get_gt_mesh(self):
    ob_name = self.videoname_to_object[self.get_video_name()]
    YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
    mesh = trimesh.load(f'{YCB_VIDEO_DIR}/models/{ob_name}/textured_simple.obj')
    return mesh


class BopBaseReader:
  def __init__(self, base_dir, zfar=np.inf, resize=1):
    self.base_dir = base_dir
    self.resize = resize
    self.dataset_name = None
    self.color_files = sorted(glob.glob(f"{self.base_dir}/rgb/*"))
    if len(self.color_files) == 0:
        self.color_files = sorted(glob.glob(f"{self.base_dir}/gray/*"))
    self.zfar = zfar

    # Read camera intrinsics from info.yml
    info_file = f'{self.base_dir}/info.yml'
    if os.path.exists(info_file):
        with open(info_file, 'r') as ff:
            info = yaml.safe_load(ff)
        self.K_table = {f'{i:06d}': np.array(info[i]['cam_K']).reshape(3, 3) for i in info}
        #self.K_table = {f'{i:06d}': np.array(info['intrinsic_matrix']) for i in range(len(self.color_files))}
        self.bop_depth_scale = info['depth_scale'] if 'depth_scale' in info else 1.0
    else:
        raise FileNotFoundError(f"info.yml not found in {self.base_dir}")

    # Read ground truth from gt.yml
    gt_file = f'{self.base_dir}/gt.yml'
    if os.path.exists(gt_file):
        with open(gt_file, 'r') as ff:
            self.scene_gt = yaml.safe_load(ff)
    else:
        self.scene_gt = None

    self.make_id_strs()


  def make_scene_ob_ids_dict(self):
    with open(f'{BOP_DIR}/{self.dataset_name}/test_targets_bop19.json','r') as ff:
      self.scene_ob_ids_dict = {}
      data = json.load(ff)
      for d in data:
        if d['scene_id']==self.get_video_id():
          id_str = f"{d['im_id']:06d}"
          if id_str not in self.scene_ob_ids_dict:
            self.scene_ob_ids_dict[id_str] = []
          self.scene_ob_ids_dict[id_str] += [d['obj_id']]*d['inst_count']


  def get_K(self, i_frame):
    K = self.K_table[self.id_strs[i_frame]]
    if self.resize != 1:
        K[:2, :2] *= self.resize
    return K


  def get_video_dir(self):
    video_id = int(self.base_dir.rstrip('/').split('/')[-1])
    return video_id

  '''
  def make_id_strs(self):
    self.id_strs = []
    for i in range(len(self.color_files)):
      name = os.path.basename(self.color_files[i]).split('.')[0]
      self.id_strs.append(name)
  '''
  def make_id_strs(self):
    self.id_strs = [os.path.basename(f).split('.')[0] for f in self.color_files]


  def get_instance_ids_in_image(self, i_frame:int):
    ob_ids = []
    if self.scene_gt is not None:
      name = int(os.path.basename(self.color_files[i_frame]).split('.')[0])
      for k in self.scene_gt[str(name)]:
        ob_ids.append(k['obj_id'])
    elif self.scene_ob_ids_dict is not None:
      return np.array(self.scene_ob_ids_dict[self.id_strs[i_frame]])
    else:
      mask_dir = os.path.dirname(self.color_files[0]).replace('rgb','mask_visib')
      id_str = self.id_strs[i_frame]
      mask_files = sorted(glob.glob(f'{mask_dir}/{id_str}_*.png'))
      ob_ids = []
      for mask_file in mask_files:
        ob_id = int(os.path.basename(mask_file).split('.')[0].split('_')[1])
        ob_ids.append(ob_id)
    ob_ids = np.asarray(ob_ids)
    return ob_ids


  def get_gt_mesh_file(self, ob_id):
    raise RuntimeError("You should override this")


  def get_color(self, i):
    """
    Reads and processes the color image at the specified index.

    This function performs the following steps:
    - Reads the color image from the file at index 'i' using the 'imageio' library.
    - If the image is grayscale (2D), it converts it to a 3-channel RGB image by replicating the grayscale values across the three channels.
    - If resizing is enabled (i.e., `self.resize` is not 1), it resizes the color image using OpenCV.

    Args:
        i (int): The index of the color image in the 'self.color_files' list to read.

    Returns:
        numpy.ndarray: The processed color image in RGB format.
    """
    color = imageio.imread(self.color_files[i])  # Read the image
    if len(color.shape) == 2:
        color = np.tile(color[..., None], (1, 1, 3))  # Convert grayscale to RGB by replicating the grayscale values
    if self.resize != 1:
        color = cv2.resize(color, dsize=None, fx=self.resize, fy=self.resize)  # Resize the color image if needed
    return color



  def get_depth(self, i):
    """
    Reads and processes the depth image corresponding to the color image at index 'i'.

    This function performs the following steps:
    - Replaces the 'rgb' or 'gray' part of the color image filename with 'depth' to find the corresponding depth image file.
    - Reads the depth image using OpenCV's `cv2.imread` function.
    - The depth values are converted from millimeters to meters by multiplying by 1e-3 and a dataset-specific scaling factor (`self.bop_depth_scale`).
    - If resizing is enabled (i.e., `self.resize` is not 1), it resizes the depth image using nearest-neighbor interpolation.
    - Depth values that are too small (<0.001 meters) or too large (beyond the defined far plane `self.zfar`) are set to zero.

    Args:
        i (int): The index of the depth image corresponding to the color image.

    Returns:
        numpy.ndarray: The processed depth image in meters.
    """
    depth_file = self.color_files[i].replace('rgb', 'depth').replace('gray', 'depth')  # Get the depth image filename
    depth = cv2.imread(depth_file, -1) * 1e-3 * self.bop_depth_scale  # Read the depth image and scale it to meters
    if self.resize != 1:
        depth = cv2.resize(depth, dsize=None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_NEAREST)  # Resize if needed
    depth[depth < 0.001] = 0  # Set very small depth values to zero
    depth[depth > self.zfar] = 0  # Set very large depth values (beyond the far plane) to zero
    return depth

  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.get_K(i))
    return xyz_map


  def get_mask(self, i_frame: int, ob_id: int, type='mask'):
    '''
    @type: mask (projected mask from whole model)
    '''
    pos = 0
    # Get the frame number from the file name (convert to integer to match the scene_gt keys)
    frame_name = int(os.path.basename(self.color_files[i_frame]).split('.')[0])

    # Debugging: Show frame number and related info
    #print(f"[DEBUG] Frame index: {i_frame}, Frame name (from color files): {frame_name}, Object ID: {ob_id}")
    #print(f"[DEBUG] Available keys in scene_gt: {self.scene_gt.keys()}")

    # Check if the current frame exists in scene_gt
    if frame_name not in self.scene_gt:
        logging.warning(f"Frame {frame_name} not found in scene_gt for object ID {ob_id}. Skipping frame.")
        return None

    # Proceed with mask fetching if the frame is available
    print(f"Frame {frame_name} found in scene_gt")
    for k in self.scene_gt[frame_name]:
        #print(f"k = {k['obj_id']}")
        if k['obj_id'] == ob_id:
            break
        pos += 1

    # Zero-pad the frame number for file names
    frame_name_padded = f'{frame_name:04d}'  # Now padded to 4 digits

    # Modify the mask file path to match your naming convention
    mask_file = f'{self.base_dir}/{type}/{frame_name_padded}.png'

    if not os.path.exists(mask_file):
        logging.info(f'{mask_file} not found')
        return None

    # Load the mask image
    mask = cv2.imread(mask_file, -1)

    # Resize mask if necessary
    if self.resize != 1:
        mask = cv2.resize(mask, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)

    return mask > 0





  def get_gt_mesh(self, ob_id:int):
    mesh_file = self.get_gt_mesh_file(ob_id)
    mesh = trimesh.load(mesh_file)
    mesh.vertices *= 1e-3
    return mesh


  def get_model_diameter(self, ob_id):
    dir = os.path.dirname(self.get_gt_mesh_file(self.ob_ids[0]))
    info_file = f'{dir}/models_info.json'
    with open(info_file,'r') as ff:
      info = json.load(ff)
    return info[str(ob_id)]['diameter']/1e3



  def get_gt_poses(self, i_frame, ob_id):
    gt_poses = []
    name = int(self.id_strs[i_frame])
    for i_k, k in enumerate(self.scene_gt[str(name)]):
      if k['obj_id']==ob_id:
        cur = np.eye(4)
        cur[:3,:3] = np.array(k['cam_R_m2c']).reshape(3,3)
        cur[:3,3] = np.array(k['cam_t_m2c'])/1e3
        gt_poses.append(cur)
    return np.asarray(gt_poses).reshape(-1,4,4)


  def get_gt_pose(self, i_frame:int, ob_id, mask=None, use_my_correction=False):
    print("THIS ONE IN BOP BASE")
    ob_in_cam = np.eye(4)
    best_iou = -np.inf
    best_gt_mask = None
    name = int(self.id_strs[i_frame])
    for i_k, k in enumerate(self.scene_gt[str(name)]):
      if k['obj_id']==ob_id:
        cur = np.eye(4)
        cur[:3,:3] = np.array(k['cam_R_m2c']).reshape(3,3)
        cur[:3,3] = np.array(k['cam_t_m2c'])/1e3
        if mask is not None:  # When multi-instance exists, use mask to determine which one
          gt_mask = cv2.imread(f'{self.base_dir}/mask_visib/{self.id_strs[i_frame]}_{i_k:06d}.png', -1).astype(bool)
          intersect = (gt_mask*mask).astype(bool)
          union = (gt_mask+mask).astype(bool)
          iou = float(intersect.sum())/union.sum()
          if iou>best_iou:
            best_iou = iou
            best_gt_mask = gt_mask
            ob_in_cam = cur
        else:
          ob_in_cam = cur
          break


    if use_my_correction:
      if 'ycb' in self.base_dir.lower() and 'train_real' in self.color_files[i_frame]:
        video_id = self.get_video_id()
        if ob_id==1:
          if video_id in [12,13,14,17,24]:
            ob_in_cam = ob_in_cam@self.symmetry_tfs[ob_id][1]
    return ob_in_cam


  def load_symmetry_tfs(self):
    """
    Loads the symmetry transformations for each object in the dataset.

    This function loads the symmetry information from the `models_info.yml` file, which contains 
    details about discrete and continuous symmetries for objects. These symmetries are used to 
    handle objects with identical appearances from multiple orientations (e.g., cylindrical objects).

    The function extracts the symmetry transformation matrices and stores them in the `symmetry_tfs` 
    attribute, which will be used for pose refinement and other operations.

    It also populates `symmetry_info_table` and `geometry_symmetry_info_table` with the symmetry data for 
    each object.

    Args:
        None

    Returns:
        None: Populates the `symmetry_tfs` and `symmetry_info_table` attributes.
    """

    dir = os.path.dirname(self.get_gt_mesh_file(self.ob_ids[0]))
    info_file = f'{dir}/models_info.yml'

    # Load the YAML file instead of JSON
    with open(info_file,'r') as ff:
        info = yaml.safe_load(ff)

    # Print available keys for debugging
    print("Available keys in models_info.yml:", list(info.keys()))

    self.symmetry_tfs = {}
    self.symmetry_info_table = {}


    for ob_id in self.ob_ids:
        # Print the current object ID for debugging
        #print(f"Current object ID: {ob_id}")

        # Convert ob_id to an integer for comparison
        if ob_id not in info:
            print(f"Object ID {ob_id} not found in models_info.yml, skipping...")
            continue

        # Access the corresponding entry in the YAML file
        self.symmetry_info_table[ob_id] = info[ob_id]  # Use integer key here
        self.symmetry_tfs[ob_id] = symmetry_tfs_from_info(info[ob_id], rot_angle_discrete=5)
    
    self.geometry_symmetry_info_table = copy.deepcopy(self.symmetry_info_table)

  
  def get_video_id(self):
    """
    Extracts and returns the video or object ID from the directory path (self.base_dir).

    This function splits the base directory path by '/', takes the last part (which represents
    the ID), and converts it into an integer.

    Example:
        If self.base_dir is 'Linemod_preprocessed/data/01', the function will return 1.

    Returns:
        int: The video or object ID as an integer.
    """
    return int(self.base_dir.split('/')[-1])


class LinemodOcclusionReader(BopBaseReader):
  def __init__(self,base_dir='/Linemod_preprocessed/data/01', zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'lmo'
    self.K = list(self.K_table.values())[0]
    self.obs_ids = [1]
    self.ob_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    self.ob_id_to_names = {
      1: 'ape',
      2: 'benchvise',
      3: 'bowl',
      4: 'camera',
      5: 'water_pour',
      6: 'cat',
      7: 'cup',
      8: 'driller',
      9: 'duck',
      10: 'eggbox',
      11: 'glue',
      12: 'holepuncher',
      13: 'iron',
      14: 'lamp',
      15: 'phone',
    }
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    # Define where ground truth mesh models are stored for LINEMOD
    return f'Linemod_preprocessed/models/obj_{ob_id:02d}.ply'


class LinemodReader(LinemodOcclusionReader):
  def __init__(self, base_dir='/Linemod_preprocessed/data/01', zfar=np.inf, split=None):
      super().__init__(base_dir, zfar=zfar)
      self.dataset_name = 'lm'
      
      # Load camera intrinsics from info.yml
      info_file = f'{self.base_dir}/info.yml'
      if os.path.exists(info_file):
          with open(info_file, 'r') as file:
              info_data = yaml.safe_load(file)
          # Assuming camera intrinsics are stored under the key 'intrinsic_matrix'
          self.K = {f'{i:06d}': np.array(info_data[i]['cam_K']).reshape(3, 3) for i in info_data}
      else:
          raise FileNotFoundError(f"Camera intrinsics file info.yml not found in {self.base_dir}")

      # Load color files based on the split (train/test)
      if split is not None:
          split_file = f'/Linemod_preprocessed/data/{self.get_video_id():02d}/{split}.txt'
          if os.path.exists(split_file):
              with open(split_file, 'r') as ff:
                  lines = ff.read().splitlines()
              self.color_files = [f'{self.base_dir}/rgb/{int(id):06d}.png' for id in lines]
              self.make_id_strs()
          else:
              raise FileNotFoundError(f"{split}.txt file not found in {self.base_dir}")

      # Exclude certain object IDs (bowl and mug)
      self.ob_ids = np.setdiff1d(np.arange(1, 16), np.array([7, 3])).tolist()
      self.load_symmetry_tfs()

  def get_gt_pose(self, i_frame: int, ob_id: int):
    """
    Fetches the ground truth pose for a given frame and object ID.
    """
    print("THIS ONE IN LINEMODE")
    # Debugging: print frame and object id info
    #print(f"[DEBUG] Accessing ground truth pose for frame {i_frame}, object ID {ob_id}")
    
    # Ensure the frame index is kept as an integer
    gt_data = self.scene_gt
    frame_index = i_frame  # already an integer from earlier
    
    # Check if the frame exists in scene_gt
    if frame_index not in gt_data:
        logging.error(f"Frame {frame_index} not found in scene_gt")
        return None
    
    # Now, loop through the objects in this frame and find the one with matching obj_id
    for obj_data in gt_data[frame_index]:
        if obj_data['obj_id'] == ob_id:
            #print(f"[DEBUG] Found matching object ID {ob_id} in frame {frame_index}")
            return obj_data  # Return the ground truth pose for this object
    
    # If we didn't find a match
    logging.error(f"Object ID {ob_id} not found in frame {frame_index}")
    return None
  

  
  def get_gt_mesh_file(self, ob_id):
      # Locate the ground truth mesh for the object
      root = self.base_dir
      while True:
          if os.path.exists(f'{root}/models'):
              mesh_dir = f'{root}/models/obj_{ob_id:02d}.ply'
              break
          else:
              #If the models directory is not found, 
              # it moves one level up in the directory tree (../) 
              # and repeats the search until it finds the models directory.
              root = os.path.abspath(f'{root}/../')
      return mesh_dir

  def get_reconstructed_mesh(self, ob_id, ref_view_dir):
      # Get the reconstructed mesh for the object
      mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
      return mesh


class YcbVideoReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'ycbv'
    self.K = list(self.K_table.values())[0]

    self.make_id_strs()

    self.ob_ids = np.arange(1,22).astype(int).tolist()
    YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
    names = sorted(os.listdir(f'{YCB_VIDEO_DIR}/models/'))
    self.ob_id_to_names = {}
    self.name_to_ob_id = {}
    for i,ob_id in enumerate(self.ob_ids):
      self.ob_id_to_names[ob_id] = names[i]
      self.name_to_ob_id[names[i]] = ob_id

    if 'BOP' not in self.base_dir:
      with open(f'{self.base_dir}/../../keyframe.txt','r') as ff:
        self.keyframe_lines = ff.read().splitlines()

    self.load_symmetry_tfs()
    for ob_id in self.ob_ids:
      if ob_id in [1,4,6,18]:   # Cylinder
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_continuous': [
              {'axis':[0,0,1], 'offset':[0,0,0]},
            ],
          'symmetries_discrete': euler_matrix(0, np.pi, 0).reshape(1,4,4).tolist(),
          }
      elif ob_id in [13]:
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_continuous': [
              {'axis':[0,0,1], 'offset':[0,0,0]},
            ],
          }
      elif ob_id in [2,3,9,21]:   # Rectangle box
        tfs = []
        for rz in [0, np.pi]:
          for rx in [0,np.pi]:
            for ry in [0,np.pi]:
              tfs.append(euler_matrix(rx, ry, rz))
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_discrete': np.asarray(tfs).reshape(-1,4,4).tolist(),
          }
      else:
        pass

  def get_gt_mesh_file(self, ob_id):
    if 'BOP' in self.base_dir:
      mesh_file = os.path.abspath(f'{self.base_dir}/../../ycbv_models/models/obj_{ob_id:06d}.ply')
    else:
      mesh_file = f'{self.base_dir}/../../ycbv_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_mesh(self, ob_id:int, get_posecnn_version=False):
    if get_posecnn_version:
      YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
      mesh = trimesh.load(f'{YCB_VIDEO_DIR}/models/{self.ob_id_to_names[ob_id]}/textured_simple.obj')
      return mesh
    mesh_file = self.get_gt_mesh_file(ob_id)
    mesh = trimesh.load(mesh_file, process=False)
    mesh.vertices *= 1e-3
    tex_file = mesh_file.replace('.ply','.png')
    if os.path.exists(tex_file):
      from PIL import Image
      im = Image.open(tex_file)
      uv = mesh.visual.uv
      material = trimesh.visual.texture.SimpleMaterial(image=im)
      color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
      mesh.visual = color_visuals
    return mesh


  def get_reconstructed_mesh(self, ob_id, ref_view_dir):
    mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
    return mesh


  def get_transform_reconstructed_to_gt_model(self, ob_id):
    out = np.eye(4)
    return out


  def get_visible_cloud(self, ob_id):
    file = os.path.abspath(f'{self.base_dir}/../../models/{self.ob_id_to_names[ob_id]}/visible_cloud.ply')
    pcd = o3d.io.read_point_cloud(file)
    return pcd


  def is_keyframe(self, i):
    color_file = self.color_files[i]
    video_id = self.get_video_id()
    frame_id = int(os.path.basename(color_file).split('.')[0])
    key = f'{video_id:04d}/{frame_id:06d}'
    return (key in self.keyframe_lines)



class TlessReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'tless'

    self.ob_ids = np.arange(1,31).astype(int).tolist()
    self.load_symmetry_tfs()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../models_cad/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_mesh(self, ob_id):
    mesh = trimesh.load(self.get_gt_mesh_file(ob_id))
    mesh.vertices *= 1e-3
    mesh = trimesh_add_pure_colored_texture(mesh, color=np.ones((3))*200)
    return mesh


class HomebrewedReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'hb'
    self.ob_ids = np.arange(1,34).astype(int).tolist()
    self.load_symmetry_tfs()
    self.make_scene_ob_ids_dict()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../hb_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_pose(self, i_frame:int, ob_id, use_my_correction=False):
    logging.info("WARN HomeBrewed doesn't have GT pose")
    return np.eye(4)



class ItoddReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'itodd'
    self.make_id_strs()

    self.ob_ids = np.arange(1,29).astype(int).tolist()
    self.load_symmetry_tfs()
    self.make_scene_ob_ids_dict()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../itodd_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


class IcbinReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'icbin'
    self.ob_ids = np.arange(1,3).astype(int).tolist()
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../icbin_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


class TudlReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'tudl'
    self.ob_ids = np.arange(1,4).astype(int).tolist()
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../tudl_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


