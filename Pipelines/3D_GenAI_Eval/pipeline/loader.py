import trimesh

class MeshLoader:
    def __init__(self, path_obj: str, path_ply: str):
        self.path_obj = path_obj
        self.path_ply = path_ply
        self.mesh_ai = None
        self.mesh_gt = None

    def load(self):
        self.mesh_ai = trimesh.load(self.path_obj)
        self.mesh_gt = trimesh.load(self.path_ply)

    def get_meshes(self):
        return self.mesh_gt, self.mesh_ai
