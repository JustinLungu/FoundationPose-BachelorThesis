#Writing our own evaluation script will give us full control over how we compare the predicted poses (linemod_res.yml) against the ground truth poses (gt.yml).
'''
Load gt.yml → Extract ground truth poses for each frame.
Load linemod_res.yml → Extract the estimated poses from FoundationPose.
Compute Evaluation Metrics → Compare the estimated poses with the ground truth using:
ADD-S metric (Average Distance of Model Points)
Translation & Rotation Errors
IoU for bounding boxes (optional)
Save and Visualize Results → Output scores, generate histograms, and log errors.
'''


import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_yaml(file_path):
    """Loads a YAML file and returns the parsed content."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def compute_add(gt_pose, pred_pose, model_points):
    """
    Computes ADD (Average Distance of Model Points) error.
    - gt_pose: 4x4 numpy array (ground truth transformation matrix)
    - pred_pose: 4x4 numpy array (predicted transformation matrix)
    - model_points: Nx3 numpy array of object model points
    """
    transformed_gt = (gt_pose[:3, :3] @ model_points.T).T + gt_pose[:3, 3]
    transformed_pred = (pred_pose[:3, :3] @ model_points.T).T + pred_pose[:3, 3]
    distances = np.linalg.norm(transformed_gt - transformed_pred, axis=1)
    return np.mean(distances)

def compute_errors(gt_pose, pred_pose):
    """Computes translation and rotation errors between ground truth and predicted pose."""
    trans_error = np.linalg.norm(gt_pose[:3, 3] - pred_pose[:3, 3])
    
    rot_gt = R.from_matrix(gt_pose[:3, :3])
    rot_pred = R.from_matrix(pred_pose[:3, :3])
    rot_error = np.rad2deg(rot_gt.inv() * rot_pred).magnitude()
    
    return trans_error, rot_error

def evaluate(linemod_res_path, gt_path, model_points):
    """Evaluates pose estimation results by comparing to ground truth."""
    linemod_res = load_yaml(linemod_res_path)
    gt_data = load_yaml(gt_path)
    
    add_errors = []
    translation_errors = []
    rotation_errors = []
    
    for frame in linemod_res:
        for obj_id in linemod_res[frame]:
            if str(frame) not in gt_data or obj_id not in [entry['obj_id'] for entry in gt_data[int(frame)]]:
                continue  # Skip if missing GT
            
            gt_pose_list = [entry for entry in gt_data[int(frame)] if entry['obj_id'] == obj_id]
            gt_pose = np.eye(4)
            gt_pose[:3, :3] = np.array(gt_pose_list[0]['cam_R_m2c']).reshape(3, 3)
            gt_pose[:3, 3] = np.array(gt_pose_list[0]['cam_t_m2c']) / 1000.0  # Convert mm to meters
            
            pred_pose = np.array(linemod_res[frame][obj_id])
            
            add_error = compute_add(gt_pose, pred_pose, model_points)
            trans_error, rot_error = compute_errors(gt_pose, pred_pose)
            
            add_errors.append(add_error)
            translation_errors.append(trans_error)
            rotation_errors.append(rot_error)
    
    return add_errors, translation_errors, rotation_errors

def plot_results(add_errors, translation_errors, rotation_errors):
    """Plots histograms of ADD, translation, and rotation errors."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(add_errors, bins=50, color='blue', alpha=0.7)
    plt.xlabel('ADD Error (m)')
    plt.ylabel('Frequency')
    plt.title('ADD Error Distribution')
    
    plt.subplot(1, 3, 2)
    plt.hist(translation_errors, bins=50, color='green', alpha=0.7)
    plt.xlabel('Translation Error (m)')
    plt.ylabel('Frequency')
    plt.title('Translation Error Distribution')
    
    plt.subplot(1, 3, 3)
    plt.hist(rotation_errors, bins=50, color='red', alpha=0.7)
    plt.xlabel('Rotation Error (degrees)')
    plt.ylabel('Frequency')
    plt.title('Rotation Error Distribution')
    
    plt.show()

if __name__ == "__main__":
    linemod_res_path = "linemod_res.yml"  # Change to actual path
    gt_path = "gt.yml"  # Change to actual path
    model_points = np.loadtxt("object_model.xyz")  # Load object 3D model points
    
    add_errors, trans_errors, rot_errors = evaluate(linemod_res_path, gt_path, model_points)
    plot_results(add_errors, trans_errors, rot_errors)
    
    print(f"Mean ADD Error: {np.mean(add_errors):.4f} m")
    print(f"Mean Translation Error: {np.mean(trans_errors):.4f} m")
    print(f"Mean Rotation Error: {np.mean(rot_errors):.4f} degrees")
