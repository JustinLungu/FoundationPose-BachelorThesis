import open3d as o3d
import numpy as np
import os

def compare_meshes(mesh1_path, mesh2_path):
    """
    Compare two meshes by computing the distance between corresponding vertices.
    This assumes both meshes have the same number of vertices and the same ordering.
    """
    # Read the meshes
    mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
    mesh2 = o3d.io.read_triangle_mesh(mesh2_path)

    # Convert vertices to NumPy arrays
    v1 = np.asarray(mesh1.vertices)
    v2 = np.asarray(mesh2.vertices)

    if v1.shape != v2.shape:
        print("[Error] The meshes do not have the same number of vertices. Cannot compare directly.")
        return

    # Calculate per-vertex distance
    differences = v2 - v1
    distances = np.linalg.norm(differences, axis=1)

    # Compute statistics
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Print out results
    print(f"Comparing: {os.path.basename(mesh1_path)} vs {os.path.basename(mesh2_path)}")
    print(f"  Number of vertices: {v1.shape[0]}")
    #On average, each vertex has moved about 0.001596 units. 
    #This is a good indicator of the overall magnitude of the noise.
    print(f"  Mean distance:      {mean_dist:.6f}")
    #This value shows how much variation there is in the movement of the vertices. 
    #A smaller standard deviation means most vertices moved close to the mean value, 
    #while a larger one would indicate a wider spread in displacements.
    print(f"  Std distance:       {std_dist:.6f}")
    #Some vertices barely move, which suggests that not all vertices are affected 
    #equally by the noise (which is expected with random Gaussian noise).
    print(f"  Min distance:       {min_dist:.6f}")
    #The largest observed displacement for a single vertex. 
    #This shows that while most vertices moved by a small amount, a few moved more noticeably.
    print(f"  Max distance:       {max_dist:.6f}")
    print("")

if __name__ == "__main__":
    # Modify these paths to point to your original and noisy models
    mesh1_path = "models/obj_01.ply"
    mesh2_path = "models_noisy/obj_01.ply"

    compare_meshes(mesh1_path, mesh2_path)
