import open3d as o3d
import numpy as np
import os
import csv

class MeshComparer:
    def __init__(self, original_folder, noisy_folder):
        self.original_folder = original_folder
        self.noisy_folder = noisy_folder

    def compare_all(self, output_csv="comparison_results.csv"):
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

            mesh1 = o3d.io.read_triangle_mesh(original_path)
            mesh2 = o3d.io.read_triangle_mesh(noisy_path)

            v1 = np.asarray(mesh1.vertices)
            v2 = np.asarray(mesh2.vertices)

            if v1.shape != v2.shape:
                print(f"[Error] Mismatched vertex count in {fname}")
                continue

            distances = np.linalg.norm(v2 - v1, axis=1)
            result = {
                "filename": fname,
                "num_vertices": v1.shape[0],
                "mean_distance": np.mean(distances),
                "std_distance": np.std(distances),
                "min_distance": np.min(distances),
                "max_distance": np.max(distances),
            }
            results.append(result)

            print(f"Compared: {fname}")
            print(f"  Mean: {result['mean_distance']:.6f}, Std: {result['std_distance']:.6f}, Min: {result['min_distance']:.6f}, Max: {result['max_distance']:.6f}")

        # Save results to CSV
        csv_path = os.path.join(self.noisy_folder, output_csv)
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"\n✅ Comparison results saved to: {csv_path}")
