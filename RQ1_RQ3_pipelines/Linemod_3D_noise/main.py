from pipeline.noise.gaussian import GaussianNoise
from pipeline.noise.normal import NormalNoise
from pipeline.noise.speckle import SpeckleNoise
from pipeline.noise.outlier import OutlierNoise
from pipeline.mesh_processor import MeshProcessor
from pipeline.comparer import MeshComparer
import os

def main():
    input_folder = "models"
    output_base_folder = "models_noisy"

    noise_types = {
        "gaussian": GaussianNoise(mean=0, std_dev=0.1),
        "normal": NormalNoise(std_dev=0.1),
        "speckle": SpeckleNoise(std_dev=0.1),
        "outlier": OutlierNoise(percentage=0.02, std_dev=0.1)
    }

    for name, noise in noise_types.items():
        print(f"\n--- Processing {name} noise ---")
        processor = MeshProcessor(input_folder, output_base_folder, noise, name)
        processor.process_all()

        comparer = MeshComparer(input_folder, os.path.join(output_base_folder, name))
        comparer.compare_all(output_csv=f"comparison_{name}.csv")

if __name__ == "__main__":
    main()