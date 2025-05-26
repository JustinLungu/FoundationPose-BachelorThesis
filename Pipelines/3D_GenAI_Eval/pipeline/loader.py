import trimesh
import numpy as np

class MeshLoader:
    def __init__(self, path_ai: str, path_gt: str):
        self.path_ai = path_ai
        self.path_gt = path_gt
        self.mesh_ai = None
        self.mesh_gt = None

    def load(self):
        def process_mesh(mesh):
            # Handle different return types from trimesh.load()
            if isinstance(mesh, trimesh.Trimesh):
                return mesh
            elif isinstance(mesh, trimesh.Scene):
                return mesh.dump().sum()
            elif isinstance(mesh, list):
                # Combine all meshes in the list
                return trimesh.util.concatenate(mesh)
            else:
                raise ValueError(f"Unknown mesh type: {type(mesh)}")

        # Load AI mesh
        ai_mesh = trimesh.load(self.path_ai)
        self.mesh_ai = process_mesh(ai_mesh)

        # Load GT mesh
        gt_mesh = trimesh.load(self.path_gt)
        self.mesh_gt = process_mesh(gt_mesh)

    def get_meshes(self):
        return self.mesh_gt, self.mesh_ai