import os
import glob
from extractions.extract_depth import DepthEstimator
from extractions.extract_masks import SegmentationMaskProcessor
from hots_process import HOTSDirectoryCreator

def get_matching_rgb(segmentation_file, rgb_dir):
    """Finds the corresponding RGB image for a given segmentation file."""
    base_name = os.path.basename(segmentation_file).replace(".npy", ".png")
    return os.path.join(rgb_dir, base_name)

def main():
    # Step 1: Create Directory Structure
    print("Creating directory structure...")
    directory_creator = HOTSDirectoryCreator(label_mapping_file="HOTS_v1/label_mapping.csv")
    directory_creator.create_structure()

    # Step 2: Get all segmentation masks and process them
    segmentation_dir = "HOTS_v1/scene/SemanticSegmentation/SegmentationClass"
    rgb_dir = "HOTS_v1/scene/RGB"
    segmentation_files = glob.glob(os.path.join(segmentation_dir, "*.npy"))

    print(f"Found {len(segmentation_files)} segmentation files.")

    for seg_file in segmentation_files:
        rgb_file = get_matching_rgb(seg_file, rgb_dir)
        if not os.path.exists(rgb_file):
            print(f"Warning: No matching RGB image found for {seg_file}, skipping.")
            continue

        print(f"Processing {seg_file} with RGB {rgb_file}...")

        processor = SegmentationMaskProcessor(
            npy_file=seg_file,
            rgb_file=rgb_file,
            label_mapping_file="HOTS_v1/label_mapping.csv"
        )
        processor.process_masks()
        processor.summary()

    # Step 3: Run depth estimation on all objects' RGB images
    print("Running depth estimation on all processed images...")
    depth_estimator = DepthEstimator()
    depth_estimator.process_all_objects(base_dir="HOTS_Processed")

    print("Full dataset processing complete!")

if __name__ == "__main__":
    main()
