from pathlib import Path
from typing import Dict
import numpy as np
import yaml

from .data_loader.loader import DataLoader
from .processing.formatter import DataFormatter
from .evaluation.core import PoseEvaluator
from .visualization import Plotter, PointCloudViewer

def load_config(config_path: Path) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def run_pipeline(config: Dict):
    # Format input files if needed
    if config.get('format_inputs', True):
        DataFormatter.reformat_gt(
            Path(config['paths']['raw_gt']),
            Path(config['paths']['ground_truth'])
        )
        DataFormatter.reformat_results(
            Path(config['paths']['raw_results']),
            Path(config['paths']['predictions'])
        )
    
    # Load data
    gt = DataLoader.load_yaml(config['paths']['ground_truth'])
    pred = DataLoader.load_yaml(config['paths']['predictions'])
    cloud = DataLoader.load_ply(config['paths']['point_cloud'])
    
    # Evaluate
    evaluator = PoseEvaluator(gt, pred, cloud)
    metrics = evaluator.evaluate()
    
    # Visualize
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    Plotter().plot_metrics(metrics, output_dir)
    PointCloudViewer().compare(gt[0].matrix, pred[0].matrix, cloud)
    
    return metrics

if __name__ == "__main__":
    config = load_config(Path("config/default.yaml"))
    results = run_pipeline(config)
    print("Evaluation complete. Metrics:", {
        k: f"{np.mean(v):.4f}" for k,v in results.items()
    })