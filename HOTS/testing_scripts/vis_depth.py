import os
import numpy as np
import matplotlib.pyplot as plt

# Define the root directory for the dataset
root_dir = './HOTS_v1'  # This will refer to HOTS_V1 folder within HOTS

# Example to access the 'scene' directory inside HOTS_V1
scene_dir = os.path.join(root_dir, 'scene')

# Example: Access a specific file like the .npy file in InstanceSegmentation
depth_dir = os.path.join(scene_dir, 'InstanceSegmentation', 'SegmentationObject')
depth_file = os.path.join(depth_dir, 'kitchen_5_top_raw_0.npy')

# Load the depth file (assuming it's a .npy file)
depth_data = np.load(depth_file)

print("Depth Data Shape:", depth_data.shape)

# Visualize the depth data
plt.imshow(depth_data, cmap='gray')
plt.colorbar()
plt.show()