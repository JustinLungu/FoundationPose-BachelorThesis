import yaml
import numpy as np
from typing import Dict, Any
from pathlib import Path
from ..data_loader.models import Transformation

class DataFormatter:
    @staticmethod
    def reformat_gt(input_path: Path, output_path: Path) -> None:
        """Reformat ground truth YAML to standard structure"""
        with open(input_path) as f:
            data = yaml.safe_load(f)
        
        formatted = {}
        for frame, objects in data.items():
            obj = objects[0]  # Assuming one object per frame
            obj_id = str(obj["obj_id"])
            
            # Create 4x4 transformation matrix
            rotation = np.array(obj["cam_R_m2c"]).reshape(3,3)
            translation = np.array(obj["cam_t_m2c"]) / 1000.0  # mm to meters
            matrix = np.eye(4)
            matrix[:3,:3] = rotation
            matrix[:3,3] = translation
            
            if obj_id not in formatted:
                formatted[obj_id] = {}
                
            formatted[obj_id][int(frame)] = {
                obj_id: matrix.tolist()
            }
        
        with open(output_path, 'w') as f:
            yaml.dump(formatted, f, default_flow_style=False)

    @staticmethod
    def reformat_results(input_path: Path, output_path: Path) -> None:
        """Standardize results YAML structure"""
        with open(input_path) as f:
            data = yaml.safe_load(f)
        
        # Simple pass-through for already formatted files
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)