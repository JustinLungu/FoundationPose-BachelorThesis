import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import DPTImageProcessor, DPTForDepthEstimation
import os

class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-large", device=None):
        """Initializes the depth estimation model."""
        # Set OpenCV to run in headless mode (fixes Qt errors)
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        
        # Check if CUDA is available
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        # Load MiDaS model
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def estimate_depth(self, image_path, output_path="depth_output.png", gamma=5.0):
        """Estimates depth from an RGB image and saves the depth map."""
        # Load RGB image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get depth prediction
        with torch.no_grad():
            depth = self.model(**inputs).predicted_depth
        
        # Move depth map back to CPU for visualization
        depth = depth.squeeze().cpu().numpy()
        
        # Normalize depth for better visualization
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = np.power(depth, gamma)  # Adjust gamma for contrast
        
        # Save depth map
        plt.imshow(depth, cmap="gray")
        plt.colorbar()
        plt.title("Depth Map")
        plt.imsave(output_path, depth, cmap="gray")
        print(f"Depth map saved as {output_path}")
        
        return depth
    
# Example Usage
if __name__ == "__main__":
    estimator = DepthEstimator()
    depth_map = estimator.estimate_depth("HOTS_v1/scene/RGB/kitchen_5_top_raw_0.png")
