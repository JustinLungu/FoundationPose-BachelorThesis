# Directory paths
BASE_DIR = "hots_data/HOTS_v1"
DEPTH_DIR = "hots_data/depth"
MESH_DIR = "hots_data/3D_models"
CAM_FILE_PATH = "hots_data/cam_K.txt"

# Output format and location
FORMAT_TYPE = "demo"  # Options: "linemod", "demo"
OUTPUT_DIR = f"../../Hots_pose_estimation/data/HOTS_Processed_{FORMAT_TYPE}"

# File names
LABEL_MAPPING_FILE = f"{BASE_DIR}/label_mapping.csv"
SEGMENTATION_DIR = f"{BASE_DIR}/scene/SemanticSegmentation/SegmentationClass"
RGB_DIR = f"{BASE_DIR}/scene/RGB"

# Mesh scaling (real-world object dimensions in meters)
TARGET_DIMS = {
    "apple": 0.08, "banana": 0.15, "book": 0.22, "bowl": 0.19, "can": 0.12,
    "cup": 0.11, "fork": 0.19, "juice_box": 0.17, "keyboard": 0.45, "knife": 0.20,
    "laptop": 0.33, "lemon": 0.08, "marker": 0.15, "milk": 0.24, "monitor": 0.33,
    "mouse": 0.11, "orange": 0.08, "peach": 0.08, "pear": 0.08, "pen": 0.15,
    "plate": 0.24, "pringles": 0.23, "scissors": 0.17, "spoon": 0.19, "stapler": 0.18
}

# Shared category mappings (to avoid duplicate base meshes)
SHARED_CATEGORIES = {
    "book": "Book", "can": "Can", "cup": "Cup", "fork": "Fork", "marker": "Marker",
    "pen": "Pen", "plate": "Plate", "pringles": "Pringles", "scissors": "Scissors", "spoon": "Spoon"
}

# Mesh rotation (in radians)
ROTATION_X = -3.14159 / 2  # -90 degrees
ROTATION_Y = 0
ROTATION_Z = 3.14159       # 180 degrees
