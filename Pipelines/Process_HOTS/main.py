"""
Main pipeline script for processing HOTS dataset into FoundationPose-compatible format.

This script:
1. Scans for segmentation mask files
2. Matches them with corresponding RGB and depth images
3. Processes each scene through the HOTSProcessorManager
4. Handles 3D mesh processing and final dataset organization
"""

from hots_pipeline.manager import HOTSProcessorManager
from hots_pipeline.config import (
    DEPTH_DIR, 
    MESH_DIR, 
    CAM_FILE_PATH, 
    FORMAT_TYPE, 
    OUTPUT_DIR, 
    LABEL_MAPPING_FILE, 
    SEGMENTATION_DIR, 
    RGB_DIR,
    SKIP_IMAGES_CONTAINING,
    REQUIRE_DEPTH_NPY
)
import glob
import os


if __name__ == "__main__":
    print(f"Searching for masks in: {SEGMENTATION_DIR}")
    mask_files = glob.glob(os.path.join(SEGMENTATION_DIR, "*.npy"))
    print(f"Found {len(mask_files)} mask files")

    all_objects = {}  # Track processed objects and counts
    processor = None  # Will be initialized with first valid file

    for i, mask_file in enumerate(mask_files, 1):
        base = os.path.splitext(os.path.basename(mask_file))[0]
        if any(keyword.lower() in base.lower() for keyword in SKIP_IMAGES_CONTAINING):
            print(f"Skipping all files for '{base}' (contains excluded keyword)")
            continue

        rgb_file = os.path.join(RGB_DIR, base + ".png")
        depth_npy_file = os.path.join(DEPTH_DIR, base + ".npy")
        depth_png_file = os.path.join(DEPTH_DIR, base + ".png")

        # Check depth file requirements
        if REQUIRE_DEPTH_NPY and not os.path.exists(depth_npy_file):
            print(f"Skipping all files for '{base}' (missing .npy depth file)")
            continue
        elif not os.path.exists(depth_npy_file) and not os.path.exists(depth_png_file):
            print(f"Skipping all files for '{base}' (missing depth file)")
            continue

        # Log current processing status
        print(f"\nProcessing file {i}/{len(mask_files)}:")
        print(f"Mask: {mask_file}")
        print(f"RGB: {rgb_file}")
        print(f"Depth: {depth_npy_file if os.path.exists(depth_npy_file) else depth_png_file}")

        if not os.path.exists(rgb_file):
            print(f"!!! RGB not found for {mask_file} !!!")
            continue

        processor = HOTSProcessorManager(
            rgb_file=rgb_file,
            mask_file=mask_file,
            label_mapping_file=LABEL_MAPPING_FILE,
            depth_dir=DEPTH_DIR,
            output_dir=OUTPUT_DIR,
            cam_file_path=CAM_FILE_PATH,
            mesh_dir=MESH_DIR,
            format_type=FORMAT_TYPE
        )
        processor.process()

        # Update object counts
        for obj, count in processor.object_counter.items():
            all_objects[obj] = all_objects.get(obj, 0) + count

    if processor is not None:
        # Process 3D meshes
        processor.finalization_3d()

        # Renumber files if in linemod format
        if FORMAT_TYPE == "linemod":
            print("\nRenaming files to sequential numbering per object...")
            processor.renumber_files_per_object(OUTPUT_DIR)

        print("\n=== Processing Summary ===")
        for obj, count in sorted(all_objects.items()):
            print(f" - {obj}: {count} image(s)")
    else:
        print("\nNo valid files were processed!")