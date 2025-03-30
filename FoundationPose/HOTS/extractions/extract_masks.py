import numpy as np
import pandas as pd
import cv2
import os
import shutil

class SegmentationMaskProcessor:
    def __init__(self, npy_file, rgb_file, label_mapping_file, output_dir="HOTS_Processed"):
        self.npy_file = npy_file
        self.rgb_file = rgb_file
        self.label_mapping_file = label_mapping_file
        self.output_dir = output_dir
        self.mask_data = None
        self.id_to_name_mapping = {}
        self.saved_masks = {}

        # Load data
        self._load_mask()
        self._load_label_mapping()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_mask(self):
        """Loads the segmentation mask from the npy file."""
        self.mask_data = np.load(self.npy_file, allow_pickle=True)

    def _load_label_mapping(self):
        """Loads the label mapping from the CSV file."""
        label_mapping_df = pd.read_csv(self.label_mapping_file)
        self.id_to_name_mapping = dict(zip(label_mapping_df["ID"], label_mapping_df["Instance"]))

    def process_masks(self):
        """Processes each unique object in the segmentation mask and saves binary masks and RGB images."""
        unique_labels = np.unique(self.mask_data)

        if 0 in unique_labels:
            unique_labels = unique_labels[unique_labels != 0]  # Remove background if 0 represents it

        # Read RGB image
        rgb_image = cv2.imread(self.rgb_file)

        # Extract filename without extension
        image_name = os.path.basename(self.rgb_file).replace(".png", "")

        for label in unique_labels:
            object_name = self.id_to_name_mapping.get(label, f"object_{label}")
            object_dir = os.path.join(self.output_dir, object_name)

            # Create subdirectories (RGB, Depth, Mask, Mesh) if not exist
            for subfolder in ["RGB", "Depth", "Mask", "Mesh"]:
                os.makedirs(os.path.join(object_dir, subfolder), exist_ok=True)

            # Save the binary mask
            binary_mask = (self.mask_data == label).astype(np.uint8) * 255
            mask_filename = os.path.join(object_dir, "Mask", f"{image_name}.png")
            cv2.imwrite(mask_filename, binary_mask)

            # Copy the RGB image to the object's RGB folder
            rgb_output_path = os.path.join(object_dir, "RGB", f"{image_name}.png")
            shutil.copy(self.rgb_file, rgb_output_path)

            self.saved_masks[label] = (mask_filename, rgb_output_path)

        print(f"Processed masks and RGB images for {len(self.saved_masks)} objects.")

    def summary(self):
        """Prints a summary of the saved masks and images."""
        for label, (mask_path, rgb_path) in self.saved_masks.items():
            object_name = self.id_to_name_mapping.get(label, f"object_{label}")
            print(f"Label {label} ({object_name}):")
            print(f"  - Mask saved at: {mask_path}")
            print(f"  - RGB saved at: {rgb_path}")

# Example Usage
if __name__ == "__main__":
    processor = SegmentationMaskProcessor(
        npy_file="HOTS_v1/scene/SemanticSegmentation/SegmentationClass/kitchen_5_top_raw_0.npy", 
        rgb_file="HOTS_v1/scene/RGB/kitchen_5_top_raw_0.png",
        label_mapping_file="HOTS_v1/label_mapping.csv"
    )
    processor.process_masks()
    processor.summary()
