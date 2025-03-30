from hots.processor_rgb import RGBProcessor
from hots.processor_depth import DepthProcessor
from hots.processor_mask import MaskProcessor
from hots.structure import HOTSDirectoryCreator
from hots.mesh_processor import HOTSMeshProcessor

import os
import numpy as np
import cv2
import pandas as pd

class HOTSProcessorManager:
    def __init__(self, rgb_file, mask_file, label_mapping_file, depth_dir, output_dir, cam_file_path, mesh_dir):
        self.rgb_file = rgb_file
        self.mask_file = mask_file
        self.depth_dir = depth_dir
        self.output_dir = output_dir
        self.cam_file_path = cam_file_path
        self.mask_data = np.load(mask_file, allow_pickle=True)
        self.rgb_image = cv2.imread(rgb_file)
        self.label_map = self._load_labels(label_mapping_file)
        self.dir_creator = HOTSDirectoryCreator(label_mapping_file, output_dir, cam_file_path)
        self.mesh_processor = HOTSMeshProcessor(source_dir=mesh_dir, target_dir=output_dir, label_mapping_file=label_mapping_file)
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
            object_dir = self.dir_creator.create_object_subfolders(object_name)

            RGBProcessor(self.rgb_file).save_to(os.path.join(object_dir, "RGB", f"{image_name}.png"))
            MaskProcessor(self.mask_data, label).save_to(image_name, os.path.join(object_dir, "Mask"))
            DepthProcessor(self.depth_dir).save_to(image_name, os.path.join(object_dir, "Depth", f"{image_name}.png"))

            self.object_counter[object_name] = self.object_counter.get(object_name, 0) + 1

        print(f"Finished processing scene '{image_name}'")

    def finalization_3d(self):
        print("\n============ Preprocessing and placing all 3D mesh models ============")
        self.mesh_processor.process_all()
        print("============ Mesh processing complete ============")