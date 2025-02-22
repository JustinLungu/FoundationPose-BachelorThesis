import numpy as np
import pandas as pd
import cv2
import os

# --- Configuration --- #
npy_file = "kitchen_5_top_raw_0.npy"  # Path to segmentation mask
label_mapping_file = "label_mapping.csv"  # Path to label mapping
output_dir = "segmented_masks"  # Directory to store individual masks

# --- Load the Segmentation Mask --- #
mask_data = np.load(npy_file, allow_pickle=True)

# Get unique object labels in the mask
unique_labels = np.unique(mask_data)
if 0 in unique_labels:
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (if 0 represents background)

# --- Load the Label Mapping --- #
label_mapping_df = pd.read_csv(label_mapping_file)

# Convert label mapping to dictionary (ID -> Object Name)
id_to_name_mapping = dict(zip(label_mapping_df["ID"], label_mapping_df["Instance"]))

# --- Create Output Directory --- #
os.makedirs(output_dir, exist_ok=True)

# --- Process Each Object --- #
saved_masks = {}

for label in unique_labels:
    object_name = id_to_name_mapping.get(label, f"object_{label}")  # Use name if available, else generic
    object_dir = os.path.join(output_dir, object_name)  # Create folder for each object
    os.makedirs(object_dir, exist_ok=True)

    # Create binary mask for the current object
    binary_mask = (mask_data == label).astype(np.uint8) * 255  # Convert to 0-255 grayscale

    # Save the mask
    mask_filename = os.path.join(object_dir, f"{object_name}.png")
    cv2.imwrite(mask_filename, binary_mask)
    
    # Store for reference
    saved_masks[label] = mask_filename

print(f"Saved {len(saved_masks)} masks in '{output_dir}'")

# --- Summary Output --- #
for label, filepath in saved_masks.items():
    object_name = id_to_name_mapping.get(label, f"object_{label}")
    print(f"Label {label} ({object_name}) mask saved at: {filepath}")
