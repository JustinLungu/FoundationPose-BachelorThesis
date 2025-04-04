import os
import json
from pipeline.constants import RESULTS_DIR, AI_DIR, GT_DIR

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


def evaluate_single_model(obj_file, ply_file):
    print(f"\n=== Evaluating {obj_file} vs {ply_file} ===")

    loader = MeshLoader(os.path.join(AI_DIR, obj_file), os.path.join(GT_DIR, ply_file))
    loader.load()
    mesh_gt, mesh_ai = loader.get_meshes()

    model_id = os.path.splitext(obj_file)[0]
    model_dir = os.path.join(RESULTS_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # Preprocessing
    preprocessor = MeshPreprocessor(mesh_gt, mesh_ai)
    preprocessor.center()

    vis = MeshVisualizer(mesh_gt, mesh_ai)
    vis.show(f"{obj_file} - Before Scaling", save_path=os.path.join(model_dir, "before_scaling.png"))
    preprocessor.safe_scaling()
    vis.show(f"{obj_file} - After Scaling", save_path=os.path.join(model_dir, "after_scaling.png"))

    # Refinement
    refiner = MeshRefiner(mesh_gt, mesh_ai)
    refiner.apply_ransac()
    refiner.apply_multiscale_icp()
    vis.show(f"{obj_file} - After ICP", save_path=os.path.join(model_dir, "after_icp.png"))

    # Metrics
    bool_iou_metric = IoUBoolMetric(mesh_gt, mesh_ai)
    bool_iou = bool_iou_metric.compute()
    bool_iou_class = bool_iou_metric.get_class(bool_iou)

    voxel_iou_metric = IoUVoxelMetric(mesh_gt, mesh_ai)
    voxel_iou = voxel_iou_metric.compute()
    voxel_iou_class = voxel_iou_metric.get_class(voxel_iou)

    chamfer_eval = ChamferMetric(mesh_gt, mesh_ai, model_dir)
    chamfer = chamfer_eval.compute()
    chamfer_class = chamfer_eval.get_class(chamfer)

    hausdorff_eval = HausdorffDistanceEvaluator(mesh_gt, mesh_ai, model_dir)
    hausdorff = hausdorff_eval.compute()
    hausdorff_class = hausdorff_eval.get_class(hausdorff)

    normal_eval = NormalConsistencyEvaluator(mesh_gt, mesh_ai, model_dir)
    normal_score = normal_eval.compute(visualize=True)
    normal_class = normal_eval.get_class(normal_score)

    curv_eval = MeanCurvatureEvaluator(mesh_gt, mesh_ai, model_dir)
    mean_curv = curv_eval.compute(visualize=True)
    curv_class = curv_eval.get_class(mean_curv)

    emd_eval = EMDEvaluator(mesh_gt, mesh_ai, model_dir)
    emd_score = emd_eval.compute(visualize=True)
    emd_class = emd_eval.get_class(emd_score)

    result = {
        'model': obj_file,
        
        'boolean_iou': {
            'score': bool_iou,
            'class': bool_iou_class  # No thresholds defined (yet)
        },
        'voxel_iou': {
            'score': voxel_iou,
            'class': voxel_iou_class
        },
        'chamfer_distance': {
            'score': chamfer,
            'class': chamfer_class
        },
        'hausdorff_distance': {
            'score': hausdorff,
            'class': hausdorff_class
        },
        'normal_consistency': {
            'score': normal_score,
            'class': normal_class
        },
        'mean_curvature_error': {
            'score': mean_curv,
            'class': curv_class
        },
        'emd': {
            'score': emd_score,
            'class': emd_class
        }
    }


    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=4)

    return result


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    obj_files = sorted([f for f in os.listdir(AI_DIR) if f.endswith('.obj')])
    ply_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith('.ply')])

    results = [evaluate_single_model(obj, ply) for obj, ply in zip(obj_files, ply_files)]

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== SUMMARY ===")
    for res in results:
        print(f"{res['model']}: "
            f"IoU = {res['boolean_iou']['score']:.4f} ({res['boolean_iou']['class']}), "
            f"Voxel IoU = {res['voxel_iou']['score']:.4f} ({res['voxel_iou']['class']}), "
            f"Hausdorff = {res['hausdorff_distance']['score']:.4f} ({res['hausdorff_distance']['class']}), "
            f"Chamfer = {res['chamfer_distance']['score']:.4f} ({res['chamfer_distance']['class']}), "
            f"Normal = {res['normal_consistency']['score']:.4f} ({res['normal_consistency']['class']}), "
            f"Curvature = {res['mean_curvature_error']['score']:.6f} ({res['mean_curvature_error']['class']}), "
            f"EMD = {res['emd']['score']:.4f} ({res['emd']['class']})")

