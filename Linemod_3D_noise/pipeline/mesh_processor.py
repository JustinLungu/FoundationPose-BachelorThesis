import os
import open3d as o3d

class MeshProcessor:
    def __init__(self, input_folder, output_folder, noise_strategy):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.noise_strategy = noise_strategy
        os.makedirs(self.output_folder, exist_ok=True)

    def process_all(self):
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.ply', '.obj')):
                input_path = os.path.join(self.input_folder, filename)
                mesh = o3d.io.read_triangle_mesh(input_path)

                if mesh.is_empty():
                    print(f"Skipping invalid mesh: {filename}")
                    continue

                noisy_mesh = self.noise_strategy.apply(mesh)
                output_path = os.path.join(self.output_folder, filename)
                o3d.io.write_triangle_mesh(output_path, noisy_mesh)
                print(f"Saved noisy mesh to: {output_path}")
