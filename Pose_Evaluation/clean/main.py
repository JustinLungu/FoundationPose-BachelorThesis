# main.py

from formatter import GroundTruthFormatter, ResultsFormatter
from ply_inspector import PointCloudInspector
from evaluator import TransformationEvaluator
from pose_visualizer import PoseVisualizer

def main():
    # Step 1: Format the ground truth YAML file
    print("Formatting ground truth YAML file...")
    gt_formatter = GroundTruthFormatter("gt.yml", "gt_reformatted.yml")
    gt_formatter.format_and_save()
    
    # Step 2: Format the results YAML file
    print("Formatting results YAML file...")
    res_formatter = ResultsFormatter("linemod_res.yml", "res_reformatted.yml")
    res_formatter.format_and_save()
    
    # Step 3: Inspect the point cloud dimensions
    print("Inspecting point cloud dimensions...")
    inspector = PointCloudInspector("obj_01.ply")
    inspector.print_dimensions()
    
    # Step 4: Evaluate transformations and generate error plots
    print("Evaluating transformation errors...")
    evaluator = TransformationEvaluator("gt_reformatted.yml", "res_reformatted.yml", "obj_01.ply")
    metrics = evaluator.evaluate()
    evaluator.plot_results()
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Step 5: Visualize the pose alignment for a specific frame (e.g., frame 0)
    print("Visualizing pose alignment for frame 0...")
    visualizer = PoseVisualizer("gt_reformatted.yml", "res_reformatted.yml", "obj_01.ply")
    visualizer.visualize_frame(frame_index=0)

if __name__ == "__main__":
    main()
