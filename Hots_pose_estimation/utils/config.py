import os
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

######### LINEMOD ##########

# === CONSTANTS ===
OBJECT_ID = 1
DEBUG_LEVEL = 5
DETECT_TYPE = 'mask'
USE_RECONSTRUCTED_MESH = 0
DEVICE = 'cuda:0'

LINEMOD_DIR = "data/HOTS_Processed_linemod/data/01"
MESH_DIR = "data/HOTS_Processed_linemod/models"
DEBUG_DIR = "results/linemod_run/apple"
REF_VIEW_DIR = "data/HOTS_Processed_linemod/ref_views"