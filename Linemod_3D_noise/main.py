from pipeline.noise.gaussian import GaussianNoise
from pipeline.mesh_processor import MeshProcessor
from pipeline.comparer import MeshComparer
import os

def main():
    input_folder = "models"
    output_folder = "models_noisy"

    # Apply Gaussian noise
    noise = GaussianNoise(mean=2, std_dev=0.1)
    processor = MeshProcessor(input_folder, output_folder, noise)
    processor.process_all()

    # Compare all models
    comparer = MeshComparer(input_folder, output_folder)
    comparer.compare_all(output_csv="comparison_results.csv")

if __name__ == "__main__":
    main()
