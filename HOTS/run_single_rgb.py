from extractions.extract_depth import DepthEstimator
from extractions.extract_masks import SegmentationMaskProcessor

# Example Usage
if __name__ == "__main__":
    estimator = DepthEstimator()
    depth_map = estimator.estimate_depth("HOTS_v1/scene/RGB/kitchen_5_top_raw_0.png")

    processor = SegmentationMaskProcessor(npy_file="HOTS_v1/scene/SemanticSegmentation/SegmentationClass/kitchen_5_top_raw_0.npy", 
                                          label_mapping_file="HOTS_v1/label_mapping.csv", 
                                          catalogue_file="HOTS_v1/catalogue.tsv")
    processor.process_masks()
    processor.summary()

