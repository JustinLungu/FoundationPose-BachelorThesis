import os
import numpy as np
import shutil
import imageio.v2 as imageio
from .base import ModalityProcessor

class DepthProcessor(ModalityProcessor):
    def __init__(self, depth_dir):
        self.depth_dir = depth_dir

    def get_depth_file(self, base_name):
        npy_path = os.path.join(self.depth_dir, f"{base_name}.npy")
        png_path = os.path.join(self.depth_dir, f"{base_name}.png")
        if os.path.exists(png_path): return png_path, "png"
        if os.path.exists(npy_path): return npy_path, "npy"
        return None, None

    def save_to(self, base_name, output_path):
        depth_file, file_type = self.get_depth_file(base_name)
        if file_type is None:
            print(f"!!! No depth file found for {base_name} !!!")
            return

        if file_type == "png":
            shutil.copy(depth_file, output_path)
        else:
            depth = np.load(depth_file)
            if depth.dtype == np.float32 or np.max(depth) <= 10:
                depth = (depth * 1000).astype(np.uint16)
            imageio.imwrite(output_path, depth)
