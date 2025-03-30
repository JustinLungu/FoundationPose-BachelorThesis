from .processor_rgb import RGBProcessor
from .processor_depth import DepthProcessor
from .processor_mask import MaskProcessor
from .structure import HOTSDirectoryCreator
from .mesh_processor import HOTSMeshProcessor

import os
import numpy as np
import cv2
import pandas as pd

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
        
        # Ensure all directories exist before saving files
        os.makedirs(os.path.join(obj_data_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(obj_data_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(obj_data_dir, "mask"), exist_ok=True)
        
        # Save RGB
        rgb_path = os.path.join(obj_data_dir, "rgb", f"{image_name}.png")
        RGBProcessor(self.rgb_file).save_to(rgb_path)
        
        # Save Depth
        depth_path = os.path.join(obj_data_dir, "depth", f"{image_name}.png")
        DepthProcessor(self.depth_dir).save_to(image_name, depth_path)
        
        # Save Mask
        MaskProcessor(self.mask_data, label).save_to(image_name, os.path.join(obj_data_dir, "mask"))

    def finalization_3d(self):
        print("\n============ Preprocessing and placing all 3D mesh models ============")
        self.mesh_processor.process_all()
        print("============ Mesh processing complete ============")