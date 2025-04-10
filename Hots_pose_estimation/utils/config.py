import os
from typing import List, Union

PIPELINE_MODE = "linemod" # "demo" or "linemod"

############### DEMO CONFIGURATION ##########
class DemoConfig:
    USE_MASK_EVERY_FRAME: bool = True
    DEBUG_LEVEL: int = 2
    ITERATION_REGISTER: int = 5
    ITERATION_TRACK: int = 2
    AXIS_SCALE: float = 0.1
    AXIS_THICKNESS: int = 3
    TRANSPARENCY: int = 0
    SKIP_FRAMES_CONTAINING: List[str] = ["kitchen"]

    DEMO_ROOT: str = "data/HOTS_Processed_demo"
    OUTPUT_ROOT: str = "results/demo_run"

    PROCESS_ALL_OBJECTS: bool = False # Set to False to only process objects in CUSTOM_OBJECT_IDS
    CUSTOM_OBJECT_IDS: List[str] = ["pringles_purple",
                                    "pringles_hot"
                                    ]  # Used only if PROCESS_ALL_OBJECTS = False


######### LINEMOD CONFIGURATION ##########
class LinemodConfig:
    PROCESS_ALL_OBJECTS: bool = True
    CUSTOM_OBJECT_IDS: List[int] = [20, 1]
    DEBUG_LEVEL: int = 0
    DETECT_TYPE: str = 'mask'
    USE_RECONSTRUCTED_MESH: int = 0
    DEVICE: str = 'cuda:0'

    LINEMOD_DIR: str = "data/HOTS_Processed_linemod/data"
    MESH_DIR: str = "data/HOTS_Processed_linemod/models"
    DEBUG_DIR: str = "results/linemod_run"
    REF_VIEW_DIR: str = "data/HOTS_Processed_linemod/ref_views"