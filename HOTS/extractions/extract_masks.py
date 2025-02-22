import numpy as np
import pandas as pd
import cv2
import os

class SegmentationMaskProcessor:
    def __init__(self, npy_file, label_mapping_file, catalogue_file=None, output_dir="segmented_masks"):
        self.npy_file = npy_file
        self.label_mapping_file = label_mapping_file
        self.catalogue_file = catalogue_file
        self.output_dir = output_dir
        self.mask_data = None
        self.id_to_name_mapping = {}
        self.saved_masks = {}
        
        # Load data
        self._load_mask()
        if not os.path.exists(self.label_mapping_file):  # Skip generation if it already exists
            self._generate_label_mapping()
        self._load_label_mapping()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_mask(self):
        """Loads the segmentation mask from the npy file."""
        self.mask_data = np.load(self.npy_file, allow_pickle=True)
    
    def _generate_label_mapping(self):
        """Generates label mapping from a catalogue file if provided."""
        if self.catalogue_file:
            catalogue_df = pd.read_csv(self.catalogue_file, sep="\t")
            label_mapping_df = catalogue_df[["ID", "Instance"]]
            label_mapping_df.to_csv(self.label_mapping_file, index=False)
            print(f"Label mapping saved to {self.label_mapping_file} inside HOTS_v1")
    
    def _load_label_mapping(self):
        """Loads the label mapping from the CSV file."""
        label_mapping_df = pd.read_csv(self.label_mapping_file)
        self.id_to_name_mapping = dict(zip(label_mapping_df["ID"], label_mapping_df["Instance"]))
    
    def process_masks(self):
        """Processes each unique object in the segmentation mask and saves binary masks."""
        unique_labels = np.unique(self.mask_data)
        
        if 0 in unique_labels:
            unique_labels = unique_labels[unique_labels != 0]  # Remove background if 0 represents it
        
        for label in unique_labels:
            object_name = self.id_to_name_mapping.get(label, f"object_{label}")
            object_dir = os.path.join(self.output_dir, object_name)
            os.makedirs(object_dir, exist_ok=True)
            
            binary_mask = (self.mask_data == label).astype(np.uint8) * 255  # Convert to 0-255 grayscale
            mask_filename = os.path.join(object_dir, f"{object_name}.png")
            cv2.imwrite(mask_filename, binary_mask)
            
            self.saved_masks[label] = mask_filename
        
        print(f"Saved {len(self.saved_masks)} masks in '{self.output_dir}'")
    
    def summary(self):
        """Prints a summary of the saved masks."""
        for label, filepath in self.saved_masks.items():
            object_name = self.id_to_name_mapping.get(label, f"object_{label}")
            print(f"Label {label} ({object_name}) mask saved at: {filepath}")

# Example Usage
if __name__ == "__main__":
    processor = SegmentationMaskProcessor(npy_file="kitchen_5_top_raw_0.npy", 
                                          label_mapping_file="label_mapping.csv", 
                                          catalogue_file="catalogue.tsv")
    processor.process_masks()
    processor.summary()
