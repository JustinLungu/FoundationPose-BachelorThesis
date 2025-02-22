import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import DPTImageProcessor, DPTForDepthEstimation
import os

class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-large", device=None):
        """Initializes the depth estimation model."""
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Load MiDaS model
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def estimate_depth(self, image_path, output_path):
        """Estimates depth from an RGB image and saves the depth map."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            depth = self.model(**inputs).predicted_depth

        depth = depth.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = np.power(depth, 5.0)

        plt.imsave(output_path, depth, cmap="gray")
        print(f"Depth map saved as {output_path}")

    def process_all_objects(self, base_dir="HOTS_Processed"):
        """Runs depth estimation on all objects' RGB images."""
        for object_name in os.listdir(base_dir):
            object_dir = os.path.join(base_dir, object_name)
            rgb_dir = os.path.join(object_dir, "RGB")
            depth_dir = os.path.join(object_dir, "Depth")

            if os.path.exists(rgb_dir):
                for rgb_file in os.listdir(rgb_dir):
                    rgb_path = os.path.join(rgb_dir, rgb_file)
                    depth_path = os.path.join(depth_dir, rgb_file)  # Matching name

                    self.estimate_depth(rgb_path, depth_path)

# Example Usage
if __name__ == "__main__":
    depth_estimator = DepthEstimator()
    depth_estimator.process_all_objects()
