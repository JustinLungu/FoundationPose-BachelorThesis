from pathlib import Path
from typing import Dict
import numpy as np
import yaml
import sys
import os

from data_loader.loader import DataLoader
from processing.formatter import DataFormatter
from evaluation.core import PoseEvaluator
from visualization.viewer import PointCloudViewer
from visualization.plotter import Plotter

# Ensure proper imports (only needed if running directly)
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_config(config_path: Path) -> Dict:
    """Load and validate configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate required paths
    required_paths = ['raw_gt', 'ground_truth', 'raw_results', 
                     'predictions', 'point_cloud', 'output_dir']
    for key in required_paths:
        if key not in config.get('paths', {}):
            raise ValueError(f"Missing required path in config: {key}")
    
    return config

def run_pipeline(config: Dict) -> Dict:
    """Main pipeline execution"""
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
    
    # Evaluate and get all metrics
    evaluator = PoseEvaluator(gt, pred, cloud)
    results = evaluator.evaluate()
    
    # Set up output directory
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Generate all visualizations
    plotter = Plotter()
    
    # 1. Save 2D plots (outliers, trends, histograms)
    plotter.plot_metrics(
        metrics=results['errors'],
        save_path=output_dir
    )
    
    # 2. Save 3D comparison
    viewer = PointCloudViewer()
    viewer.compare(
        gt[0].matrix, 
        pred[0].matrix, 
        cloud,
        save_path=output_dir / "3d_alignment.png"
    )
    
    return results['metrics']

if __name__ == "__main__":
    try:
        config = load_config(Path("../config/default.yml"))
        results = run_pipeline(config)
        
        print("\nEvaluation complete. Metrics:")
        for metric, value in results.items():
            print(f"{metric:<25}: {value:.4f}")
            
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        sys.exit(1)