import os
import shutil
import pandas as pd
import yaml

class HOTSDirectoryCreator:
    def __init__(self, label_mapping_file, output_dir, cam_file_path, format_type):
        self.label_mapping_file = label_mapping_file
        self.output_dir = output_dir
        self.cam_file_path = cam_file_path
        self.format_type = format_type
        self.id_to_name_mapping = {}
        self.name_to_id_mapping = {}
        
        self._load_label_mapping()
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_label_mapping(self):
        df = pd.read_csv(self.label_mapping_file)
        self.id_to_name_mapping = dict(zip(df["ID"], df["Instance"]))
        self.name_to_id_mapping = {v: k for k, v in self.id_to_name_mapping.items()}

    def create_structure(self):
        if self.format_type == "demo":
            for object_name in self.id_to_name_mapping.values():
                self.create_demo_object_subfolders(object_name)
        else:
            self.create_linemod_structure()

    def create_demo_object_subfolders(self, object_name):
        object_dir = os.path.join(self.output_dir, object_name)
        os.makedirs(object_dir, exist_ok=True)

        for subfolder in ["RGB", "Depth", "Mask", "Mesh"]:
            os.makedirs(os.path.join(object_dir, subfolder), exist_ok=True)

        cam_dest = os.path.join(object_dir, "cam_K.txt")
        if not os.path.exists(cam_dest):
            shutil.copy2(self.cam_file_path, cam_dest)

        return object_dir

    def create_linemod_structure(self):
        models_dir = os.path.join(self.output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)

        for object_name, object_id in self.name_to_id_mapping.items():
            obj_id_str = f"{object_id:02d}"
            obj_data_dir = os.path.join(self.output_dir, "data", obj_id_str)
            
            for subfolder in ["rgb", "depth", "mask"]:
                os.makedirs(os.path.join(obj_data_dir, subfolder), exist_ok=True)
            
            self._create_info_yml(obj_data_dir)
            self._create_gt_yml(obj_data_dir, object_id)

    def _create_info_yml(self, obj_data_dir):
        with open(self.cam_file_path, 'r') as f:
            cam_data = [float(x) for x in f.read().split()]
        
        info_data = {
            0: {
                "cam_K": cam_data,
                "depth_scale": 1.0
            }
        }
        
        with open(os.path.join(obj_data_dir, "info.yml"), 'w') as f:
            yaml.dump(info_data, f)

    def _create_gt_yml(self, obj_data_dir, obj_id):
        gt_data = {
            0: {
                "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "cam_t_m2c": [0.0, 0.0, 100.0],
                "obj_id": obj_id
            }
        }
        
        with open(os.path.join(obj_data_dir, "gt.yml"), 'w') as f:
            yaml.dump(gt_data, f)