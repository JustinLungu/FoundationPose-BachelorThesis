import numpy as np
from evaluation import TransformationEvaluator
from visualizer import TransformationVisualizer, AlignmentVisualizer
from formatter import YAMLFormatter

if __name__ == "__main__":
    formatter = YAMLFormatter()

    # Reformat input files
    formatter.reformat_predictions("data/linemod_res.yml", "reformatted/res_reformatted.yml")
    formatter.reformat_ground_truth("data/gt.yml", "reformatted/gt_reformatted.yml")

    # Evaluate errors
    evaluator = TransformationEvaluator(
        "reformatted/gt_reformatted.yml",
        "reformatted/res_reformatted.yml",
        "data/obj_01.ply"
    )
    errors = evaluator.evaluate()

    # Print average metrics
    for metric, value in {k: np.mean(v) for k, v in errors.items()}.items():
        print(f"{metric}: {value:.4f}")

    # Set frame index
    frame_idx = 0

    # Initialize visualizer
    visualizer_3d = AlignmentVisualizer(
        "reformatted/gt_reformatted.yml",
        "reformatted/res_reformatted.yml",
        "data/obj_01.ply"
    )

    # Save zoomed view (even tighter zoom)
    visualizer_3d.save_alignment_image(
        output_path=f"plots/frame_{frame_idx}_zoomed.png",
        frame_index=frame_idx,
        zoom_factor=0.3  # <- tighter zoom
    )

    # Save full view
    visualizer_3d.save_alignment_image(
        output_path=f"plots/frame_{frame_idx}_zoomed.png",
        frame_index=frame_idx,
        zoom_factor=0.4
    )

    # Annotate the zoomed view
    visualizer_3d.save_annotated_image(
        base_img_path=f"plots/frame_{frame_idx}_zoomed.png",
        output_path=f"plots/frame_{frame_idx}_annotated.png",
        frame_index=frame_idx,
        errors=errors
    )

    visualizer_3d.save_orbit_gif(
        frame_index=0,
        output_path="plots/orbit_animation.gif",
        zoom_factor=0.5  # lower = zoom in more
    )

    # Plot errors
    visualizer = TransformationVisualizer(
        errors["Rotation Error (deg)"],
        errors["Translation Error (m)"],
        errors["Pose Error (Frobenius norm)"],
        errors["ADD (m)"]
    )
    visualizer.plot_outliers()
    visualizer.plot_trends()
    visualizer.plot_distributions()
