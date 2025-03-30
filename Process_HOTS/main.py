from hots_pipeline.manager import HOTSProcessorManager
import glob
import os

BASE_DIR = "hots_data/HOTS_v1"
OUTPUT_DIR = "HOTS_Processed"
DEPTH_DIR = "hots_data/depth"
MESH_DIR = "hots_data/3D_models"
CAM_FILE_PATH = "hots_data/cam_K.txt"

def main():
    segmentation_dir = os.path.join(BASE_DIR, "scene/SemanticSegmentation/SegmentationClass")
    rgb_dir = os.path.join(BASE_DIR, "scene/RGB")
    label_mapping_file = os.path.join(BASE_DIR, "label_mapping.csv")
    mask_files = glob.glob(os.path.join(segmentation_dir, "*.npy"))
    all_objects = {}

    for mask_file in mask_files:
        base = os.path.splitext(os.path.basename(mask_file))[0]
        rgb_file = os.path.join(rgb_dir, base + ".png")

        if not os.path.exists(rgb_file):
            print(f"!!! RGB not found for {mask_file} !!!")
            continue

        processor = HOTSProcessorManager(
            rgb_file=rgb_file,
            mask_file=mask_file,
            label_mapping_file=label_mapping_file,
            depth_dir=DEPTH_DIR,
            output_dir=OUTPUT_DIR,
            cam_file_path=CAM_FILE_PATH,
            mesh_dir=MESH_DIR
        )
        processor.process()
        
        # merge the counts
        for obj, count in processor.object_counter.items():
            all_objects[obj] = all_objects.get(obj, 0) + count
    
    processor.finalization_3d()
    print("\nObject processing summary:")
    for obj, count in sorted(all_objects.items()):
        print(f" - {obj}: {count} image(s)")

if __name__ == "__main__":
    main()