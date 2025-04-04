import open3d as o3d
import numpy as np
import os
import csv

class MeshComparer:
    def __init__(self, original_folder, noisy_folder):
        self.original_folder = original_folder
        self.noisy_folder = noisy_folder

    def compare_all(self, output_csv="comparison_results.csv"):
        """
        Compare all corresponding mesh files in the original and noisy folders
        and save the statistics to a CSV file.
        """
        results = []
        filenames = sorted([
            f for f in os.listdir(self.original_folder)
            if f.lower().endswith('.ply')
        ])

        for fname in filenames:
            original_path = os.path.join(self.original_folder, fname)
            noisy_path = os.path.join(self.noisy_folder, fname)

            if not os.path.exists(noisy_path):
                print(f"[Warning] Missing noisy file for: {fname}")
                continue

            # Read both meshes
            mesh1 = o3d.io.read_triangle_mesh(original_path)
            mesh2 = o3d.io.read_triangle_mesh(noisy_path)

            # Convert vertices to NumPy arrays
            v1 = np.asarray(mesh1.vertices)
            v2 = np.asarray(mesh2.vertices)

            # Ensure both meshes are structurally compatible
            if v1.shape != v2.shape:
                print(f"[Error] Mismatched vertex count in {fname}")
                continue

            # Calculate per-vertex distances
            differences = v2 - v1
            distances = np.linalg.norm(differences, axis=1)

            # Compute descriptive statistics:
            # On average, each vertex has moved about X units.
            # This gives a general idea of the noise magnitude.
            mean_dist = np.mean(distances)

            # Shows how much variation there is in vertex movement.
            # A smaller value = more uniform displacement.
            std_dist = np.std(distances)

            # Indicates that some vertices barely moved at all.
            # As expected from Gaussian noise.
            min_dist = np.min(distances)

            # This shows the largest movement observed.
            # Some vertices might have moved significantly more.
            max_dist = np.max(distances)

            # Store results
            result = {
                "filename": fname,
                "num_vertices": v1.shape[0],
                "mean_distance": mean_dist,
                "std_distance": std_dist,
                "min_distance": min_dist,
                "max_distance": max_dist,
            }
            results.append(result)

        # Save results to CSV
        csv_path = os.path.join(self.noisy_folder, output_csv)
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"\nComparison results saved to: {csv_path}")
