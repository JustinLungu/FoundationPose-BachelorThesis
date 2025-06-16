import matplotlib
matplotlib.use("Agg")
import os
import json
from pipeline.config import RESULTS_DIR, AI_DIR, GT_DIRS, DEFAULT_OFFSET, ENABLE_VISUALIZATION
from pipeline.loader import MeshLoader
from pipeline.preprocessing import MeshPreprocessor
from pipeline.visualizer import MeshVisualizer
from pipeline.refiner import MeshRefiner
from pipeline.metrics.iou import IoUBoolMetric, IoUVoxelMetric
from pipeline.metrics.chamfer import ChamferMetric
from pipeline.metrics.hausdorff import HausdorffDistanceEvaluator
from pipeline.metrics.normal_consistency import NormalConsistencyEvaluator
from pipeline.metrics.mean_curvature_error import MeanCurvatureEvaluator
from pipeline.metrics.emd import EMDEvaluator

def find_matching_pairs():
    """Find matching AI and GT models, checking both GT directories."""
    pairs = []
    
    # Walk through all AI models
    for root, dirs, files in os.walk(AI_DIR):
        for file in files:
            if file.endswith('.obj'):
                ai_path = os.path.join(root, file)
                model_name = os.path.splitext(file)[0]
                
                # First try to find in internet GT (.obj)
                gt_path = find_ground_truth(model_name, GT_DIRS['internet'], '.obj')
                mode = 'internet'
                
                # If not found, try linemod GT (.ply)
                if not gt_path:
                    gt_path = find_ground_truth(model_name, GT_DIRS['linemod'], '.ply')
                    mode = 'linemod'
                
                if gt_path:
                    pairs.append({
                        'ai_path': ai_path,
                        'gt_path': gt_path,
                        'category': os.path.basename(os.path.dirname(root)),
                        'method': os.path.basename(os.path.dirname(os.path.dirname(root))),
                        'time': os.path.basename(root),
                        'mode': mode
                    })
                else:
                    print(f"Warning: No GT found for {model_name}")
    return pairs

def find_ground_truth(model_name, gt_dir, ext):
    """Find matching ground truth file with flexible naming."""
    # First try exact match in root directory
    possible_names = [
        f"{model_name}{ext}",                 # exact match (camera.obj)
        f"{model_name.split('_')[0]}{ext}",   # banana for banana_10mins
        f"{model_name.replace('_', '')}{ext}", # remove underscores
        f"{model_name.lower()}{ext}",         # lowercase version
        f"{model_name.split('_')[0].lower()}{ext}"  # lowercase first part
    ]
    
    # Check root directory first
    for name in possible_names:
        gt_path = os.path.join(gt_dir, name)
        if os.path.exists(gt_path):
            return gt_path
    
    # If in internet_gt_models, check subdirectories
    if 'internet' in gt_dir:
        for root, dirs, files in os.walk(gt_dir):
            for file in files:
                if file.endswith(ext):
                    # Check if model_name matches directory or file name
                    dir_name = os.path.basename(root).lower()
                    file_base = os.path.splitext(file)[0].lower()
                    model_lower = model_name.lower()
                    
                    if (model_lower in dir_name or 
                        model_lower in file_base or 
                        model_lower.split('_')[0] in dir_name):
                        return os.path.join(root, file)
    
    return None

def evaluate_single_model(pair_info):
    """Process a single model pair with robust error handling."""
    print(f"\n=== Evaluating {pair_info['ai_path']} vs {pair_info['gt_path']} ===")
    
    # Create results directory
    result_dir = os.path.join(
        RESULTS_DIR,
        pair_info['method'],
        pair_info['category'],
        pair_info['time'],
        os.path.splitext(os.path.basename(pair_info['ai_path']))[0]
    )
    os.makedirs(result_dir, exist_ok=True)

    # Initialize result structure
    result = {
        'metadata': {
            'method': pair_info['method'],
            'category': pair_info['category'],
            'time': pair_info['time'],
            'gt_source': pair_info['mode'],
            'ai_model': os.path.basename(pair_info['ai_path']),
            'gt_model': os.path.basename(pair_info['gt_path']),
            'scale_factor': None
        },
        'metrics': {}
    }

    try:
        # Load meshes
        loader = MeshLoader(pair_info['ai_path'], pair_info['gt_path'])
        loader.load()
        mesh_gt, mesh_ai = loader.get_meshes()

        # Preprocessing
        preprocessor = MeshPreprocessor(mesh_gt, mesh_ai)
        preprocessor.center()
        vis = MeshVisualizer(mesh_gt, mesh_ai)
        vis.show("Before Scaling", save_path=os.path.join(result_dir, "before_scaling.png"))
        
        scale_factor = preprocessor.safe_scaling()
        result['metadata']['scale_factor'] = scale_factor
        vis.show("After Scaling", save_path=os.path.join(result_dir, "after_scaling.png"))

        # Refinement
        refiner = MeshRefiner(mesh_gt, mesh_ai)
        mesh_ai = refiner.apply_ransac_icp()
        #refiner.apply_multiscale_icp()
        vis.show("After RANSAC+ICP", save_path=os.path.join(result_dir, "after_ransac_icp.png"))

        # Compute metrics with individual error handling
        metric_functions = {
            #'boolean_iou': lambda: IoUBoolMetric(mesh_gt, mesh_ai),
            #'voxel_iou': lambda: IoUVoxelMetric(mesh_gt, mesh_ai, slice_batch_size=1),
            'chamfer_distance': lambda: ChamferMetric(mesh_gt, mesh_ai, result_dir),
            'hausdorff_distance': lambda: HausdorffDistanceEvaluator(mesh_gt, mesh_ai, result_dir),
            'normal_consistency': lambda: NormalConsistencyEvaluator(mesh_gt, mesh_ai, result_dir),
            'mean_curvature_error': lambda: MeanCurvatureEvaluator(mesh_gt, mesh_ai, result_dir),
            'emd': lambda: EMDEvaluator(mesh_gt, mesh_ai, result_dir)
        }

        for name, metric_fn in metric_functions.items():
            try:
                metric = metric_fn()
                print(f"Computing {name}...")
                score = metric.compute(visualize=ENABLE_VISUALIZATION)
                print(f"{name} computed: {score}")
                result['metrics'][name] = {
                    'score': score,
                    'class': metric.get_class(score) if hasattr(metric, 'get_class') else 'unknown'
                }
            except Exception as e:
                print(f"Error computing {name}: {e}")
                result['metrics'][name] = {
                    'score': None,
                    'class': 'error',
                    'error': str(e)
                }
        print("Metrics computation complete.")

    except Exception as e:
        print(f"Critical error in evaluation: {e}")
        result['error'] = str(e)

    # Always save results, even if partial
    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=4)

    return result

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    matched_pairs = find_matching_pairs()
    if not matched_pairs:
        raise ValueError("No matching pairs found between AI and GT directories")

    results = [evaluate_single_model(pair) for pair in matched_pairs]

    # Save comprehensive summary
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== SUMMARY ===")
    for res in results:
        meta = res['metadata']
        metrics = res['metrics']
        print(f"{meta['method']}/{meta['category']}/{meta['time']}/{meta['ai_model']} vs {meta['gt_model']} ({meta['gt_source']}):")
        print(f"  IoU={metrics['boolean_iou']['score']:.3f}, Chamfer={metrics['chamfer_distance']['score']:.1f}, Hausdorff={metrics['hausdorff_distance']['score']:.1f}")