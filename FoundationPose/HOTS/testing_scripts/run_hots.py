import sys
import os

# Add the directory containing estimater.py to Python's sys.path
# Adjust this path to the directory where 'estimater.py' is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hots import load_HOTS_scenes
from estimater import FoundationPose
from Utils import *
import yaml

def run_pose_estimation_worker_hots(est, img, target, debug):
    # Use the RGB image and instance mask for pose estimation
    ob_mask = target["instance_masks"]
    
    # Estimating pose (Adapting FoundationPose.register for HOTS)
    pose = est.register(K=None, rgb=img, depth=None, ob_mask=ob_mask, ob_id=None)
    
    if debug >= 1:
        print(f"Pose estimated: {pose}")
    
    return pose

def run_pose_estimation_hots():
    wp.force_load(device='cuda')

    # Load the HOTS dataset
    train_data, _ = load_HOTS_scenes(root=opt.hots_dir, transform=True)

    # Initialize FoundationPose for pose estimation
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()  # Dummy mesh initialization
    est = FoundationPose(
        model_pts=mesh_tmp.vertices.copy(),
        model_normals=mesh_tmp.vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=mesh_tmp,
        scorer=None,
        refiner=None,
        glctx=glctx,
        debug_dir=opt.debug_dir,
        debug=opt.debug
    )

    res = NestDict()

    # Loop through the HOTS dataset (this is similar to how Linemod and YCB work with ob_ids)
    for i in range(len(train_data)):
        img, target = train_data[i]
        
        # Call worker function to process each image and estimate the pose
        pose = run_pose_estimation_worker_hots(est, img, target, opt.debug)
        
        # Save the results
        res[i]["pose"] = pose

    # Save results to a YAML file
    with open(f'{opt.debug_dir}/hots_res.yml', 'w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Specify HOTS dataset directory
    parser.add_argument('--hots_dir', type=str, default="HOTS/HOTS_v1", help="HOTS dataset root directory")
    parser.add_argument('--debug', type=int, default=0, help="Debug level (0 for no debug)")
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug', help="Directory to save debug information")
    
    opt = parser.parse_args()

    # Set the random seed for reproducibility (optional but recommended)
    set_seed(0)

    # Run the HOTS-specific pose estimation function
    run_pose_estimation_hots()