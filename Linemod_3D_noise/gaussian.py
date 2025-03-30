import os
import open3d as o3d
import numpy as np

def add_gaussian_noise(mesh, mean=0.0, std_dev=0.001):
    """
    Add Gaussian noise to the vertices of the mesh.
    :param mesh: open3d.geometry.TriangleMesh object
    :param mean: Mean of the Gaussian distribution
    :param std_dev: Standard deviation of the Gaussian distribution
    :return: Mesh with noisy vertices
    """
    # Convert the mesh vertices to a NumPy array
    vertices = np.asarray(mesh.vertices)

    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, vertices.shape)

    # Add noise to the vertices
    vertices_noisy = vertices + noise

    # Update the mesh with the noisy vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices_noisy)
    return mesh

def process_models(input_folder, output_folder, mean=0.0, std_dev=0.001):
    """
    Load each 3D model from input_folder, add Gaussian noise, and save it to output_folder.
    :param input_folder: Folder containing original models
    :param output_folder: Folder to save noisy models
    :param mean: Mean of the Gaussian noise
    :param std_dev: Standard deviation of the Gaussian noise
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.ply') or filename.lower().endswith('.obj'):
            file_path = os.path.join(input_folder, filename)
            
            # Read the mesh using Open3D
            mesh = o3d.io.read_triangle_mesh(file_path)
            
            # Skip empty or invalid meshes
            if mesh.is_empty():
                print(f"Skipping empty/invalid mesh: {filename}")
                continue
            
            # Add Gaussian noise
            noisy_mesh = add_gaussian_noise(mesh, mean, std_dev)

            # Construct the output path and save
            output_path = os.path.join(output_folder, filename)
            o3d.io.write_triangle_mesh(output_path, noisy_mesh)
            print(f"Saved noisy mesh to: {output_path}")

if __name__ == "__main__":
    # Example usage:
    input_folder = "models"
    output_folder = "models_noisy"

    # Adjust the standard deviation as needed
    process_models(input_folder, output_folder, mean=2, std_dev=0.1)
