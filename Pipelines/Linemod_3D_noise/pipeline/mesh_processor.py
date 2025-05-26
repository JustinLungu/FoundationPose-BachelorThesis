import os
import open3d as o3d

class MeshProcessor:
    def __init__(self, input_folder, output_base_folder, noise_strategy, noise_name):
        """Initialize processor with I/O paths and noise strategy.
        Args:
            input_folder: Path to clean meshes
            output_base_folder: Root output directory
            noise_strategy: Concrete BaseNoise implementation
            noise_name: Subfolder name for this noise type
        """
        self.input_folder = input_folder
        self.output_folder = os.path.join(output_base_folder, noise_name)
        self.noise_strategy = noise_strategy
        os.makedirs(self.output_folder, exist_ok=True)

    def process_all(self):
        """Process all valid mesh files in input directory."""
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.ply', '.obj')):
                self._process_single_file(filename)

    def _process_single_file(self, filename):
        """Handle loading, processing, and saving of a single mesh file."""
        input_path = os.path.join(self.input_folder, filename)
        try:
            mesh = o3d.io.read_triangle_mesh(input_path)
            if mesh.is_empty():
                print(f"Skipping invalid mesh: {filename}")
                return

            noisy_mesh = self.noise_strategy.apply(mesh)
            output_path = os.path.join(self.output_folder, filename)
            o3d.io.write_triangle_mesh(output_path, noisy_mesh)
            print(f"Saved noisy mesh to: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")