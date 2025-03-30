import open3d as o3d
import pandas as pd
import numpy as np
import os
import shutil
import trimesh

class HOTSMeshProcessor:
    def __init__(self, source_dir, target_dir, label_mapping_file, format_type="demo"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.label_mapping_file = label_mapping_file
        self.format_type = format_type
        self.all_objects = self._load_all_objects()
        self.name_to_id_mapping = self._create_name_to_id_mapping()

        # Keep original scaling and category definitions
        self.target_dims = {
            "apple": 0.08, "banana": 0.15, "book": 0.22, "bowl": 0.19, "can": 0.12, "cup": 0.11,
            "fork": 0.19, "juice_box": 0.17, "keyboard": 0.45, "knife": 0.20, "laptop": 0.33,
            "lemon": 0.08, "marker": 0.15, "milk": 0.24, "monitor": 0.33, "mouse": 0.11,
            "orange": 0.08, "peach": 0.08, "pear": 0.08, "pen": 0.15, "plate": 0.24,
            "pringles": 0.23, "scissors": 0.17, "spoon": 0.19, "stapler": 0.18
        }
        
        self.shared_categories = {
            "book": "Book", "can": "Can", "cup": "Cup", "fork": "Fork", "marker": "Marker",
            "pen": "Pen", "plate": "Plate", "pringles": "Pringles", "scissors": "Scissors", "spoon": "Spoon"
        }
    
    def _load_all_objects(self):
        df = pd.read_csv(self.label_mapping_file)
        return df["Instance"].tolist()
        
    def _create_name_to_id_mapping(self):
        df = pd.read_csv(self.label_mapping_file)
        return dict(zip(df["Instance"], df["ID"]))

    def preprocess_and_save_mesh(self, source_obj_path, target_obj_path, category):
        if not os.path.exists(source_obj_path):
            print(f"NOT FOUND: !!! Mesh for category '{category}', skipping.")
            return False

        mesh = o3d.io.read_triangle_mesh(source_obj_path, enable_post_processing=True)
        mesh.compute_vertex_normals()

        mesh.translate(-mesh.get_center())
        R_align = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, np.pi))
        mesh.rotate(R_align, center=(0, 0, 0))

        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_dim = np.max(extent)
        if max_dim == 0:
            print(f"WARNING: !!! Mesh for category '{category}' has zero extent, skipping.")
            return False

        target_max_dim = self.target_dims.get(category, 0.1)
        scale_factor = target_max_dim / max_dim
        mesh.scale(scale_factor, center=(0, 0, 0))
        
        # Only clear textures for linemod format
        if self.format_type == "linemod":
            mesh.textures = []

        os.makedirs(os.path.dirname(target_obj_path), exist_ok=True)
        o3d.io.write_triangle_mesh(target_obj_path, mesh)
        print(f"Mesh for category '{category}' saved to: {target_obj_path}")
        return True

    def process_all(self):
        for obj_name in self.all_objects:
            category = obj_name
            for cat_prefix, folder in self.shared_categories.items():
                if obj_name.startswith(cat_prefix):
                    category = cat_prefix
                    break

            model_folder = os.path.join(self.source_dir, self.shared_categories.get(category, category).capitalize())
            input_obj_path = os.path.join(model_folder, "model.obj")
            
            if self.format_type == "demo":
                self._process_for_demo(obj_name, category, model_folder)
            else:
                self._process_for_linemod(obj_name, input_obj_path, category)

    def _process_for_demo(self, obj_name, category, model_folder):
        input_obj_path = os.path.join(model_folder, "model.obj")
        input_mtl_path = os.path.join(model_folder, "model.mtl")
        input_tex_path = os.path.join(model_folder, "texture_kd.png")

        object_mesh_dir = os.path.join(self.target_dir, obj_name, "Mesh")
        output_obj_path = os.path.join(object_mesh_dir, "model.obj")

        model_0_path = os.path.join(object_mesh_dir, "model_0.png")
        if os.path.exists(model_0_path):
            os.remove(model_0_path)
        self.preprocess_and_save_mesh(input_obj_path, output_obj_path, category)

        for src_path, name in [(input_mtl_path, "model.mtl"), (input_tex_path, "texture_kd.png")]:
            dst_path = os.path.join(object_mesh_dir, name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"NOT FOUND: !!! {name} for category '{category}', skipping.")

        if os.path.exists(os.path.join(object_mesh_dir, "model_0.png")):
            os.remove(os.path.join(object_mesh_dir, "model_0.png"))
            print(f"WARNING: Removed unexpected model_0.png in {object_mesh_dir}")

    def _process_for_linemod(self, obj_name, input_obj_path, category):
        """New Linemod format processing logic"""
        obj_id = self.name_to_id_mapping[obj_name]
        obj_id_str = f"{obj_id:02d}"
        models_dir = os.path.join(self.target_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        temp_obj_path = os.path.join(models_dir, f"temp_{obj_id_str}.obj")
        
        try:
            if self.preprocess_and_save_mesh(input_obj_path, temp_obj_path, category):
                ply_path = os.path.join(models_dir, f"obj_{obj_id_str}.ply")
                self._convert_obj_to_ply(temp_obj_path, ply_path)
        finally:
            if os.path.exists(temp_obj_path):
                os.remove(temp_obj_path)

    def _convert_obj_to_ply(self, obj_path, ply_path):
        """Convert OBJ to PLY with millimeter scaling"""
        try:
            mesh = trimesh.load(obj_path, force='mesh')
            mesh.vertices *= 1000.0  # Convert to millimeters
            mesh.export(ply_path)
            print(f"Converted and exported '{obj_path}' to '{ply_path}' with mm scaling.")
        except Exception as e:
            print(f"ERROR converting {obj_path} to PLY: {str(e)}")