from .processor_rgb import RGBProcessor
from .processor_depth import DepthProcessor
from .processor_mask import MaskProcessor
from .structure import HOTSDirectoryCreator
from .mesh_processor import HOTSMeshProcessor
import os
import numpy as np
import cv2
import pandas as pd
import yaml  # Added yaml import
import re

class HOTSProcessorManager:
    def __init__(self, rgb_file, mask_file, label_mapping_file, depth_dir, output_dir, cam_file_path, mesh_dir, format_type="demo"):
        self.rgb_file = rgb_file
        self.mask_file = mask_file
        self.depth_dir = depth_dir
        self.output_dir = output_dir
        self.cam_file_path = cam_file_path
        self.format_type = format_type
        self.mask_data = np.load(mask_file, allow_pickle=True)
        self.rgb_image = cv2.imread(rgb_file)
        self.label_map = self._load_labels(label_mapping_file)
        self.dir_creator = HOTSDirectoryCreator(label_mapping_file, output_dir, cam_file_path, format_type)
        self.mesh_processor = HOTSMeshProcessor(source_dir=mesh_dir, target_dir=output_dir, 
                                              label_mapping_file=label_mapping_file, format_type=format_type)
        self.object_counter = {}

    def _load_labels(self, label_mapping_file):
        df = pd.read_csv(label_mapping_file)
        return dict(zip(df["ID"], df["Instance"]))

    def process(self):
        image_name = os.path.splitext(os.path.basename(self.rgb_file))[0]
        labels = np.unique(self.mask_data)
        if 0 in labels:
            labels = labels[labels != 0]

        for label in labels:
            object_name = self.label_map.get(label, f"object_{label}")
            
            if self.format_type == "demo":
                self._process_for_demo(image_name, object_name, label)
            else:
                self._process_for_linemod(image_name, object_name, label)

            self.object_counter[object_name] = self.object_counter.get(object_name, 0) + 1

        print(f"Finished processing scene '{image_name}'")

    def _process_for_demo(self, image_name, object_name, label):
        object_dir = self.dir_creator.create_demo_object_subfolders(object_name)
        
        RGBProcessor(self.rgb_file).save_to(os.path.join(object_dir, "RGB", f"{image_name}.png"))
        MaskProcessor(self.mask_data, label).save_to(image_name, os.path.join(object_dir, "Mask"))
        DepthProcessor(self.depth_dir).save_to(image_name, os.path.join(object_dir, "Depth", f"{image_name}.png"))

    def _process_for_linemod(self, image_name, object_name, label):
        object_id = self.dir_creator.name_to_id_mapping[object_name]
        obj_id_str = f"{object_id:02d}"
        obj_data_dir = os.path.join(self.output_dir, "data", obj_id_str)
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(obj_data_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(obj_data_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(obj_data_dir, "mask"), exist_ok=True)
        
        # Count existing images to determine next index
        rgb_dir = os.path.join(obj_data_dir, "rgb")
        num_images = len([f for f in os.listdir(rgb_dir) if f.endswith('.png')]) if os.path.exists(rgb_dir) else 0
        
        # Save files with sequential names
        seq_num = f"{num_images:04d}"
        rgb_path = os.path.join(rgb_dir, f"{seq_num}.png")
        depth_path = os.path.join(obj_data_dir, "depth", f"{seq_num}.png")
        mask_path = os.path.join(obj_data_dir, "mask", f"{seq_num}.png")
        
        # Process files
        RGBProcessor(self.rgb_file).save_to(rgb_path)
        DepthProcessor(self.depth_dir).save_to(image_name, depth_path)
        MaskProcessor(self.mask_data, label).save_to(image_name, os.path.join(obj_data_dir, "mask"))
        
        # Update YAML files with new entry
        self._update_yaml_files(obj_data_dir, object_id, num_images)

    def _update_yaml_files(self, obj_data_dir, obj_id, image_index):
        """Update YAML files with new entry for current image"""
        # Update info.yml
        info_path = os.path.join(obj_data_dir, "info.yml")
        with open(self.cam_file_path, 'r') as f:
            cam_data = [float(x) for x in f.read().split()]
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info_data = yaml.safe_load(f) or {}
        else:
            info_data = {}
        
        # Format camera matrix with specific decimal places
        formatted_cam = [
            round(float(cam_data[0]), 4),  # 572.4114
            0.0,
            round(float(cam_data[2]), 4),  # 325.2611
            0.0,
            round(float(cam_data[4]), 5),  # 573.57043
            round(float(cam_data[5]), 5),  # 242.04899
            0.0,
            0.0,
            1.0
        ]
        
        info_data[image_index] = {
            "cam_K": formatted_cam,
            "depth_scale": 1.0
        }
        
        with open(info_path, 'w') as f:
            # Custom YAML dumping for compact array format
            yaml.dump(info_data, f, default_flow_style=None, sort_keys=False)
            # Manually adjust the formatting
            with open(info_path, 'r') as f:
                content = f.read()
            pattern = re.compile(r'cam_K:\n((?:\s+-\s+[^\n]+\n)+)')
            matches = pattern.finditer(content)

            for match in matches:
                list_block = match.group(1)
                # Extract values from each "- val" line
                values = [line.strip().lstrip('- ').strip() for line in list_block.strip().splitlines()]
                inline = f"cam_K: [{', '.join(values)}]"
                content = content.replace(f"cam_K:\n{list_block}", inline)
            with open(info_path, 'w') as f:
                f.write(content)
        
        # Update gt.yml (unchanged from previous correct version)
        gt_path = os.path.join(obj_data_dir, "gt.yml")
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gt_data = yaml.safe_load(f) or {}
        else:
            gt_data = {}
        
        gt_data[image_index] = [{
            "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "cam_t_m2c": [0.0, 0.0, 100.0],
            "obj_id": obj_id
        }]
        
        with open(gt_path, 'w') as f:
            yaml.dump(gt_data, f, default_flow_style=None)


    def finalization_3d(self):
        print("\n============ Preprocessing and placing all 3D mesh models ============")
        self.mesh_processor.process_all()
        print("============ Mesh processing complete ============")