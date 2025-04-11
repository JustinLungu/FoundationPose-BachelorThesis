from pipeline.noise.gaussian import GaussianNoise
from pipeline.noise.normal import NormalNoise
from pipeline.noise.speckle import SpeckleNoise
from pipeline.noise.outlier import OutlierNoise
from pipeline.mesh_processor import MeshProcessor
from pipeline.comparer import MeshComparer
import os

INPUT_FOLDER = "models"
OUTPUT_FOLDER = "models_noisy"

# Noise configurations with descriptive parameters
NOISE_TYPES = {
    "gaussian": GaussianNoise(mean=0, std_dev=0.1),      # Mild uniform noise
    "normal": NormalNoise(std_dev=0.1),                  # Surface-normal aligned
    "speckle": SpeckleNoise(std_dev=0.1),                # Multiplicative noise  
    "outlier": OutlierNoise(percentage=0.02, std_dev=0.1) # Sparse strong outliers
}

if __name__ == "__main__":
    for name, noise in NOISE_TYPES.items():
        print(f"\n--- Processing {name} noise ---")
        processor = MeshProcessor(INPUT_FOLDER, OUTPUT_FOLDER, noise, name)
        processor.process_all()

        comparer = MeshComparer(INPUT_FOLDER, os.path.join(OUTPUT_FOLDER, name))
        comparer.compare_all(output_csv=f"comparison_{name}.csv")