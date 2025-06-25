"""
Main execution script for pose evaluation pipeline.
Now supports method selection from config with new file naming convention.
"""

import numpy as np
import os
import pandas as pd
from pipeline.evaluation import TransformationEvaluator
from pipeline.visualizer import TransformationVisualizer, AlignmentVisualizer
from pipeline.formatter import YAMLFormatter
import pipeline.config as cfg

def ensure_dir(path):
    """Ensure output directory exists"""
    os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    # Initialize results storage
    all_results = []
    formatter = YAMLFormatter()

    for obj_id in cfg.OBJECT_IDS:
        # Prepare paths
        gt_path = f"{cfg.LINEMOD_ROOT}/ground_truth_pose/linemod_{obj_id:02d}.yml"
        ply_path = f"{cfg.LINEMOD_ROOT}/original_models/obj_{obj_id:02d}.ply"
        
        for method in cfg.POSE_METHODS:
            # Construct the new filename pattern
            pred_path = f"{cfg.LINEMOD_ROOT}/pose_estimations/{method}_obj_{obj_id:02d}_linemod_res.yml"
            
            if not os.path.exists(pred_path):
                print(f"[!] Skipping missing: {pred_path}")
                continue

            # Create output directory for this method-object pair
            output_prefix = f"plots/{method}_obj_{obj_id}"
            ensure_dir(output_prefix)

            # 1. Data Preparation
            formatted_gt = f"reformatted/gt_obj_{obj_id}.yml"
            formatted_pred = f"reformatted/{method}_obj_{obj_id}_pred.yml"
            
            formatter.reformat_ground_truth(gt_path, formatted_gt)
            formatter.reformat_predictions(pred_path, formatted_pred)

            # 2. Pose Evaluation
            evaluator = TransformationEvaluator(formatted_gt, formatted_pred, ply_path)
            errors = evaluator.evaluate()

            # Store mean errors for summary
            mean_errors = {k: np.mean(v) for k, v in errors.items()}
            mean_errors.update({"Method": method, "Object": obj_id})
            all_results.append(mean_errors)

            # Print current results
            print(f"\nResults for {method} on object {obj_id}:")
            for metric, value in mean_errors.items():
                if metric not in ["Method", "Object"]:
                    print(f"{metric}: {value:.4f}")

            # 3. Visualization - Generate all output artifacts
            visualizer_3d = AlignmentVisualizer(formatted_gt, formatted_pred, ply_path, cfg.ROTATION_ANGLES)

            # Update output paths for this method-object pair
            zoomed_img = f"{output_prefix}/frame_{cfg.FRAME_IDX}_zoomed.png"
            full_img = f"{output_prefix}/frame_{cfg.FRAME_IDX}_full.png"
            annotated_img = f"{output_prefix}/frame_{cfg.FRAME_IDX}_annotated.png"
            gif_path = f"{output_prefix}/orbit_animation.gif"
            outlier_plot = f"{output_prefix}/error_outliers.png"
            trend_plot = f"{output_prefix}/error_trends.png"
            dist_plot = f"{output_prefix}/error_distributions.png"

            # Generate all visualizations
            if cfg.SHOW_INTERACTIVE:
                visualizer_3d.show_interactive(frame_index=cfg.FRAME_IDX)
            visualizer_3d.save_alignment_image(zoomed_img, cfg.FRAME_IDX, zoom_factor=cfg.ZOOMED_ZOOM_FACTOR)
            visualizer_3d.save_alignment_image(full_img, cfg.FRAME_IDX, zoom_factor=cfg.FULL_ZOOM_FACTOR)
            visualizer_3d.save_annotated_image(zoomed_img, annotated_img, cfg.FRAME_IDX, errors)
            visualizer_3d.save_orbit_gif(cfg.FRAME_IDX, gif_path, zoom_factor=cfg.GIF_ZOOM_FACTOR)

            # Generate statistical plots
            visualizer = TransformationVisualizer(
                errors["Rotation Error (deg)"],
                errors["Translation Error (m)"],
                errors["Pose Error (Frobenius norm)"],
                errors["ADD (m)"]
            )
            visualizer.plot_outliers(outlier_plot)
            visualizer.plot_trends(trend_plot)
            visualizer.plot_distributions(dist_plot)

    # Save consolidated results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("plots/results_summary.csv", index=False)
        print("\n[✓] Saved consolidated results to plots/results_summary.csv")