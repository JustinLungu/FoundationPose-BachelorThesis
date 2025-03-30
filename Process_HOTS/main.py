# main.py
from hots_pipeline.manager import HOTSProcessorManager
import glob
import os

BASE_DIR = "hots_data/HOTS_v1"
DEPTH_DIR = "hots_data/depth"
MESH_DIR = "hots_data/3D_models"
CAM_FILE_PATH = "hots_data/cam_K.txt"

FORMAT_TYPE = "linemod"  # or "demo"

OUTPUT_DIR = f"../FoundationPose/HOTS_Processed_{FORMAT_TYPE}"

if __name__ == "__main__":
    segmentation_dir = os.path.join(BASE_DIR, "scene/SemanticSegmentation/SegmentationClass")
    rgb_dir = os.path.join(BASE_DIR, "scene/RGB")
    label_mapping_file = os.path.join(BASE_DIR, "label_mapping.csv")

    print(f"Searching for masks in: {segmentation_dir}")
    mask_files = glob.glob(os.path.join(segmentation_dir, "*.npy"))
    print(f"Found {len(mask_files)} mask files")

    all_objects = {}
    processor = None

    for i, mask_file in enumerate(mask_files, 1):
        base = os.path.splitext(os.path.basename(mask_file))[0]
        rgb_file = os.path.join(rgb_dir, base + ".png")

        print(f"\nProcessing file {i}/{len(mask_files)}:")
        print(f"Mask: {mask_file}")
        print(f"RGB: {rgb_file}")

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
            mesh_dir=MESH_DIR,
            format_type=FORMAT_TYPE
        )
        processor.process()
        
        for obj, count in processor.object_counter.items():
            all_objects[obj] = all_objects.get(obj, 0) + count

    if processor is not None:
        processor.finalization_3d()
        print("\n=== Processing Summary ===")
        for obj, count in sorted(all_objects.items()):
            print(f" - {obj}: {count} image(s)")
    else:
        print("\nNo valid files were processed!")