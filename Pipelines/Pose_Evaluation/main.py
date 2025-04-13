"""
Main execution script for pose evaluation pipeline.
Orchestrates data formatting, error computation, and visualization generation.
"""


import numpy as np
from pipeline.evaluation import TransformationEvaluator
from pipeline.visualizer import TransformationVisualizer, AlignmentVisualizer
from pipeline.formatter import YAMLFormatter
import pipeline.config as cfg

if __name__ == "__main__":

    # 1. Data Preparation - Reformat input YAMLs to standardized structure
    formatter = YAMLFormatter()
    formatter.reformat_predictions(cfg.RAW_PRED_YAML, cfg.PRED_YAML)
    formatter.reformat_ground_truth(cfg.RAW_GT_YAML, cfg.GT_YAML)

    # 2. Pose Evaluation - Compute rotation/translation/ADD errors
    evaluator = TransformationEvaluator(cfg.GT_YAML, cfg.PRED_YAML, cfg.PLY_PATH)
    errors = evaluator.evaluate()

    for metric, value in {k: np.mean(v) for k, v in errors.items()}.items():
        print(f"{metric}: {value:.4f}")

    # 3. Visualization - Generate all output artifacts
    visualizer_3d = AlignmentVisualizer(cfg.GT_YAML, cfg.PRED_YAML, cfg.PLY_PATH, cfg.ROTATION_ANGLES)

    # 3.1 Interactive 3D view (blocks execution)
    visualizer_3d.show_interactive(frame_index=cfg.FRAME_IDX)
    
    # 3.2 Static images (zoomed and full views)
    visualizer_3d.save_alignment_image(
        output_path=cfg.ZOOMED_IMG_PATH,
        frame_index=cfg.FRAME_IDX,
        zoom_factor=cfg.ZOOMED_ZOOM_FACTOR
    )
    visualizer_3d.save_alignment_image(
        output_path=cfg.FULL_IMG_PATH, 
        frame_index=cfg.FRAME_IDX,
        zoom_factor=cfg.FULL_ZOOM_FACTOR
    )
    
    # 3.3 Annotated image with error metrics
    visualizer_3d.save_annotated_image(
        base_img_path=cfg.ZOOMED_IMG_PATH,
        output_path=cfg.ANNOTATED_IMG_PATH,
        frame_index=cfg.FRAME_IDX,
        errors=errors
    )
    
    # 3.4 Orbiting GIF animation
    visualizer_3d.save_orbit_gif(
        frame_index=cfg.FRAME_IDX,
        output_path=cfg.GIF_PATH,
        zoom_factor=cfg.GIF_ZOOM_FACTOR
    )
    
    # 4. Error Analysis - Generate statistical plots
    visualizer = TransformationVisualizer(
        errors["Rotation Error (deg)"],
        errors["Translation Error (m)"],
        errors["Pose Error (Frobenius norm)"],
        errors["ADD (m)"]
    )
    visualizer.plot_outliers(cfg.OUTLIER_PLOT_PATH)
    visualizer.plot_trends(cfg.TREND_PLOT_PATH)
    visualizer.plot_distributions(cfg.DISTRIBUTION_PLOT_PATH)