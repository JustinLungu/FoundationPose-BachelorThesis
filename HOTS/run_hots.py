from extractions.extract_depth import DepthEstimator
from extractions.extract_masks import SegmentationMaskProcessor
from hots_process import HOTSDirectoryCreator

def main():
    # Step 1: Create Directory Structure
    print("Creating directory structure...")
    directory_creator = HOTSDirectoryCreator(label_mapping_file="HOTS_v1/label_mapping.csv")
    directory_creator.create_structure()

    # Step 2: Process Segmentation Masks & Save RGB Images
    print("Processing segmentation masks and saving RGB images...")
    processor = SegmentationMaskProcessor(
        npy_file="HOTS_v1/scene/SemanticSegmentation/SegmentationClass/kitchen_5_top_raw_0.npy",
        rgb_file="HOTS_v1/scene/RGB/kitchen_5_top_raw_0.png",
        label_mapping_file="HOTS_v1/label_mapping.csv"
    )
    processor.process_masks()
    processor.summary()

    # Step 3: Process Depth Maps
    print("Running depth estimation...")
    depth_estimator = DepthEstimator()
    depth_estimator.process_all_objects(base_dir="HOTS_Processed")

    print("Processing complete!")

if __name__ == "__main__":
    main()
