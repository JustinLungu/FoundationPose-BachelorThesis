import numpy as np

from evaluation import TransformationEvaluator
from visualizer import TransformationVisualizer, AlignmentVisualizer
from formatter import YAMLFormatter

if __name__ == "__main__":
    formatter = YAMLFormatter()

    # Reformat the result file (res)
    formatter.reformat_predictions("data/linemod_res.yml", "reformatted/res_reformatted.yml")

    # Reformat the ground truth file (gt)
    formatter.reformat_ground_truth("data/gt.yml", "reformatted/gt_reformatted.yml")
    evaluator = TransformationEvaluator("reformatted/gt_reformatted.yml", "reformatted/res_reformatted.yml", "data/obj_01.ply")
    errors = evaluator.evaluate()

    results = {metric: np.mean(values) for metric, values in errors.items()}
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    visualizer_3d = AlignmentVisualizer(
        "reformatted/gt_reformatted.yml",
        "reformatted/res_reformatted.yml",
        "data/obj_01.ply"
    )
    visualizer_3d.visualize()
    visualizer_3d.save_alignment_image("plots/alignmet_vis_eval.png")

    visualizer = TransformationVisualizer(
        errors["Rotation Error (deg)"],
        errors["Translation Error (m)"],
        errors["Pose Error (Frobenius norm)"],
        errors["ADD (m)"]
    )
    visualizer.plot_outliers()
    visualizer.plot_trends()
    visualizer.plot_distributions()

    