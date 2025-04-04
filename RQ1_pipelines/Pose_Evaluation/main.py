import numpy as np
from pipeline.evaluation import TransformationEvaluator
from pipeline.visualizer import TransformationVisualizer, AlignmentVisualizer
from pipeline.formatter import YAMLFormatter
import pipeline.config as cfg

if __name__ == "__main__":
    formatter = YAMLFormatter()

    formatter.reformat_predictions("data/linemod_res.yml", cfg.PRED_YAML)
    formatter.reformat_ground_truth("data/gt.yml", cfg.GT_YAML)

    evaluator = TransformationEvaluator(cfg.GT_YAML, cfg.PRED_YAML, cfg.PLY_PATH)
    errors = evaluator.evaluate()

    for metric, value in {k: np.mean(v) for k, v in errors.items()}.items():
        print(f"{metric}: {value:.4f}")

    visualizer_3d = AlignmentVisualizer(cfg.GT_YAML, cfg.PRED_YAML, cfg.PLY_PATH, cfg.ROTATION_ANGLES)

    visualizer_3d.show_interactive(frame_index=cfg.FRAME_IDX)

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

    visualizer_3d.save_annotated_image(
        base_img_path=cfg.ZOOMED_IMG_PATH,
        output_path=cfg.ANNOTATED_IMG_PATH,
        frame_index=cfg.FRAME_IDX,
        errors=errors
    )

    visualizer_3d.save_orbit_gif(
        frame_index=cfg.FRAME_IDX,
        output_path=cfg.GIF_PATH,
        zoom_factor=cfg.GIF_ZOOM_FACTOR
    )

    visualizer = TransformationVisualizer(
        errors["Rotation Error (deg)"],
        errors["Translation Error (m)"],
        errors["Pose Error (Frobenius norm)"],
        errors["ADD (m)"]
    )

    visualizer.plot_outliers(cfg.OUTLIER_PLOT_PATH)
    visualizer.plot_trends(cfg.TREND_PLOT_PATH)
    visualizer.plot_distributions(cfg.DISTRIBUTION_PLOT_PATH)
