DEFAULT_OFFSET = [50, 0, 0]
MIN_VOLUME_THRESHOLD = 1e-6
DEFAULT_NUM_SAMPLES = 5000
HIGH_RES_SAMPLES = 10000
DEFAULT_VOXEL_PITCH = 6
ZERO_TOLERANCE = 1e-6
ENABLE_VISUALIZATION = False
RESULTS_DIR = "results"
AI_DIR = 'ai_models'
GT_DIR = 'gt_models'

IOU_THRESHOLDS = {
    "excellent": 0.9,
    "good": 0.75,
    "warning": 0.5
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