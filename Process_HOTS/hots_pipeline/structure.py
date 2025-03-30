import os
import shutil
import pandas as pd

class HOTSDirectoryCreator:
    def __init__(self, label_mapping_file, output_dir, cam_file_path):
        self.label_mapping_file = label_mapping_file
        self.output_dir = output_dir
        self.cam_file_path = cam_file_path
        self.id_to_name_mapping = {}

        self._load_label_mapping()
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_label_mapping(self):
        df = pd.read_csv(self.label_mapping_file)
        self.id_to_name_mapping = dict(zip(df["ID"], df["Instance"]))

    def create_structure(self):
        for object_name in self.id_to_name_mapping.values():
            self.create_object_subfolders(object_name)

    def create_object_subfolders(self, object_name):
        object_dir = os.path.join(self.output_dir, object_name)
        os.makedirs(object_dir, exist_ok=True)

        for subfolder in ["RGB", "Depth", "Mask", "Mesh"]:
            os.makedirs(os.path.join(object_dir, subfolder), exist_ok=True)

        cam_dest = os.path.join(object_dir, "cam_K.txt")
        if not os.path.exists(cam_dest):
            shutil.copy2(self.cam_file_path, cam_dest)

        return object_dir
