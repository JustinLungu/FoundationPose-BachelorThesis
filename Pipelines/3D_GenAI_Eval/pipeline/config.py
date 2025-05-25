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

# Directory structure for input/output management
RESULTS_DIR = "results"   
AI_DIR = '3d_data/ai_models'
GT_DIR = '3d_data/gt_models'

# Volume validation and scaling parameters
MIN_VOLUME_THRESHOLD = 1e-6      # Minimum valid volume (1mm³ cube) for volume-based scaling
ZERO_TOLERANCE = 1e-6            # Floating-point comparison threshold

# Initial alignment offset (used when centering fails)
DEFAULT_OFFSET = [50, 0, 0]      # [x,y,z] offset in millimeters

# Point cloud sampling for registration and metrics
DEFAULT_NUM_SAMPLES = 5000       # Standard sampling density for most operations
HIGH_RES_SAMPLES = 10000         # High-res sampling for final ICP refinement

# Voxel grid parameters
DEFAULT_VOXEL_PITCH = 6          # Base voxel size in millimeters for voxel-based metrics


ENABLE_VISUALIZATION = False     # Set True for interactive visualization during processing
                                 # (Note: May slow down batch processing)

# --------------------------
# Volumetric Similarity
# --------------------------
IOU_THRESHOLDS = {
    "excellent": 0.9,    # >90% volume overlap → Nearly perfect match
    "good": 0.75,        # >75% → Minor deviations acceptable
    "warning": 0.5       # <50% → Significant structural differences
}

# --------------------------
# Surface Distance Metrics
# --------------------------
CHAMFER_THRESHOLDS = {   # Average point-to-surface distance (mm)
    "good": 50,          # <50mm → High quality reconstruction
    "warn": 150,         # 50-150mm → Noticeable surface deviations
    "bad": 500           # >500mm → Major reconstruction failures
}

HAUSDORFF_THRESHOLDS = {  # Maximum surface deviation (mm)
    "good": 25.0,         # <25mm → No extreme outliers
    "decent": 100.0       # 25-100mm → Some localized large errors
}


# ==============================================
# Metric Classification Thresholds
# ==============================================
# --------------------------
# Geometric Feature Metrics
# --------------------------
NORMAL_CONSISTENCY_THRESHOLDS = {  # Cosine similarity between normals (1=perfect)
    "excellent": 0.95,   # >0.95 → Nearly identical surface orientation
    "good": 0.85,        # >0.85 → Generally consistent normals
    "decent": 0.70       # <0.70 → Significant normal field distortion
}

MEAN_CURVATURE_THRESHOLDS = {  # Absolute curvature difference (mm⁻¹)
    "excellent": 0.005,  # <0.005 → Nearly identical surface curvature
    "good": 0.01,        # <0.01 → Minor curvature variations
    "warning": 0.02      # >0.02 → Noticeable curvature mismatches
}

# --------------------------
# Point Distribution Metrics
# --------------------------
EMD_THRESHOLDS = {       # Earth Mover's Distance (normalized 0-1)
    "excellent": 0.05,   # <0.05 → Nearly identical point distributions
    "good": 0.1,         # <0.1 → Generally similar distributions
    "warning": 0.2       # >0.2 → Significant distribution differences
}