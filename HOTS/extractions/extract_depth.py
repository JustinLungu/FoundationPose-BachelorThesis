import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import DPTImageProcessor, DPTForDepthEstimation
import os

# Set OpenCV to run in headless mode (fixes Qt errors)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Check if CUDA is available (use GPU if possible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load MiDaS model
model_name = "Intel/dpt-large"
processor = DPTImageProcessor.from_pretrained(model_name)  # Updated from feature_extractor
model = DPTForDepthEstimation.from_pretrained(model_name)
model.to(device)  # Move model to GPU (if available)
model.eval()

# Load RGB image
image_path = "HOTS_v1/scene/RGB/kitchen_5_top_raw_0.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess image
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to GPU

# Get depth prediction
with torch.no_grad():
    depth = model(**inputs).predicted_depth

# Move depth map back to CPU for visualization
depth = depth.squeeze().cpu().numpy()

# Normalize depth for better visualization
depth = (depth - depth.min()) / (depth.max() - depth.min())
depth = np.power(depth, 5.0)  # Increase gamma (higher = darker)

# Show depth map
plt.imshow(depth, cmap="gray")  # Simulates real depth sensor output
plt.colorbar()
plt.title("Depth Map")
print("Depth map generation complete!")
plt.imsave("depth_output.png", depth, cmap="gray")
print("Depth map saved as depth_output.png")