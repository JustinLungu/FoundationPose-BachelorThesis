# structure.py
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

    def create_directory_structure(self):
        """Public method to create the complete directory structure"""
        if self.format_type == "demo":
            self._create_demo_structure()
        else:
            self._create_linemod_structure()

    def _create_demo_structure(self):
        """Create demo format directory structure"""
        for object_name in self.id_to_name_mapping.values():
            object_dir = os.path.join(self.output_dir, object_name)
            os.makedirs(object_dir, exist_ok=True)

            for subfolder in ["rgb", "depth", "masks", "mesh"]:
                os.makedirs(os.path.join(object_dir, subfolder), exist_ok=True)

            cam_dest = os.path.join(object_dir, "cam_K.txt")
            if not os.path.exists(cam_dest):
                shutil.copy2(self.cam_file_path, cam_dest)

    def _create_linemod_structure(self):
        """Create linemod format directory structure"""
        models_dir = os.path.join(self.output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)

        for object_name, object_id in self.name_to_id_mapping.items():
            obj_id_str = f"{object_id:02d}"
            obj_data_dir = os.path.join(self.output_dir, "data", obj_id_str)
            
            for subfolder in ["rgb", "depth", "mask"]:
                os.makedirs(os.path.join(obj_data_dir, subfolder), exist_ok=True)
            
            # Initialize empty YAML files
            self._initialize_yaml_files(obj_data_dir)

    def _initialize_yaml_files(self, obj_data_dir):
        """Initialize empty YAML files for linemod format"""
        with open(os.path.join(obj_data_dir, "info.yml"), 'w') as f:
            yaml.dump({}, f)
        with open(os.path.join(obj_data_dir, "gt.yml"), 'w') as f:
            yaml.dump({}, f)

    def get_linemod_object_dir(self, object_id):
        """Get directory path for a specific object in linemod format"""
        obj_id_str = f"{object_id:02d}"
        return os.path.join(self.output_dir, "data", obj_id_str)

    def get_next_sequence_number(self, obj_data_dir):
        """Get next sequence number for files in object directory"""
        rgb_dir = os.path.join(obj_data_dir, "rgb")
        num_images = len([f for f in os.listdir(rgb_dir) if f.endswith('.png')]) if os.path.exists(rgb_dir) else 0
        return f"{num_images:04d}"