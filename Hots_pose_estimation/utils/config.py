############### DEMO ##########

# === CONSTANTS ===
USE_MASK_EVERY_FRAME = True
DEBUG_LEVEL = 2
ITERATION_REGISTER = 5
ITERATION_TRACK = 2
AXIS_SCALE = 0.1
AXIS_THICKNESS = 3
TRANSPARENCY = 0

DEMO_ROOT = "data/HOTS_Processed_demo"
OUTPUT_ROOT = "results/demo_run"

# Add these new configuration options
PROCESS_ALL_OBJECTS = True  # Set to False to only process objects in CUSTOM_OBJECT_IDS
CUSTOM_OBJECT_IDS = ["apple", "banana"]  # Used only if PROCESS_ALL_OBJECTS = False

######### LINEMOD ##########

# === CONSTANTS ===
PROCESS_ALL_OBJECTS = False  # Set to True to process all object IDs
CUSTOM_OBJECT_IDS = [20, 39]  # Used only if PROCESS_ALL_OBJECTS = False
DEBUG_LEVEL = 0
DETECT_TYPE = 'mask'
USE_RECONSTRUCTED_MESH = 0
DEVICE = 'cuda:0'

LINEMOD_DIR = "data/HOTS_Processed_linemod/data"
MESH_DIR = "data/HOTS_Processed_linemod/models"
DEBUG_DIR = "results/linemod_run"
REF_VIEW_DIR = "data/HOTS_Processed_linemod/ref_views"
