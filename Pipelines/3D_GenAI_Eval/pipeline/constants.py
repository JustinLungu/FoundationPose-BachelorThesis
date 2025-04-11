"""
Global configuration and thresholds.

Contains:
- Directory paths
- Sampling parameters
- Metric classification thresholds
- Visualization settings

Threshold Notes:
- All linear measures in millimeters
- Angular measures in radians
- Scores normalized where applicable
"""

DEFAULT_OFFSET = [50, 0, 0]
# Volume validation cutoff (1mm³ cube)
MIN_VOLUME_THRESHOLD = 1e-6
DEFAULT_NUM_SAMPLES = 5000
HIGH_RES_SAMPLES = 10000
DEFAULT_VOXEL_PITCH = 6
ZERO_TOLERANCE = 1e-6
ENABLE_VISUALIZATION = False
RESULTS_DIR = "results"
AI_DIR = '3d_data/ai_models'
GT_DIR = '3d_data/gt_models'

# Metric classification tiers (key: threshold_value)
IOU_THRESHOLDS = {
    "excellent": 0.9,    # >90% overlap
    "good": 0.75,        # >75%
    "warning": 0.5       # ≤50% requires investigation
}

CHAMFER_THRESHOLDS = {
    "good": 50,
    "warn": 150,
    "bad": 500
}

HAUSDORFF_THRESHOLDS = {
    "good": 25.0,
    "decent": 100.0
}

NORMAL_CONSISTENCY_THRESHOLDS = {
    "excellent": 0.95,   # > 0.95 → excellent local alignment
    "good":      0.85,   # > 0.85 → good enough
    "decent":   0.70    # ≤ 0.70 → might indicate poor local match
}


MEAN_CURVATURE_THRESHOLDS = {
    "excellent": 0.005,
    "good": 0.01,
    "warning": 0.02
}

EMD_THRESHOLDS = {
    "excellent": 0.05,
    "good": 0.1,
    "warning": 0.2
}