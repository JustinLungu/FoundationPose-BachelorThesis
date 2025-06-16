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
AI_DIR = '3d_data/gen_ai_models'
GT_DIRS = {
    'internet': '3d_data/internet_gt_models',  # .obj files
    'linemod': '3d_data/linemod_gt_models'    # .ply files
}

# Volume validation and scaling parameters
MIN_VOLUME_THRESHOLD = 1e-6      # Minimum valid volume (1mm³ cube) for volume-based scaling
ZERO_TOLERANCE = 1e-6            # Floating-point comparison threshold

VOXEL_SIZE = 7.0                  # Default voxel size for point cloud operations (in mm)

# Initial alignment offset (used when centering fails)
DEFAULT_OFFSET = [50, 0, 0]      # [x,y,z] offset in millimeters

# Point cloud sampling for registration and metrics
DEFAULT_NUM_SAMPLES = 30000       # Standard sampling density for most operations
HIGH_RES_SAMPLES = 10000         # High-res sampling for final ICP refinement
EMD_NUM_SAMPLES = 2000              # Number of points for Earth Mover's Distance metric

# Voxel grid parameters
DEFAULT_VOXEL_PITCH = 7          # Base voxel size in millimeters for voxel-based metrics


ENABLE_VISUALIZATION = True     # Set True for interactive visualization during processing
                                 # (Note: May slow down batch processing)

# --------------------------
# Volumetric Similarity (IoU)
# --------------------------
# References:
#  • Choy et al., “3D‐R2N2: A unified approach for single‐ and multi‐view 3D object reconstruction,” CVPR 2017
#  • Wu et al., “Learning a probabilistic latent space of object shapes via 3D‐GAN,” ICCV 2016
IOU_THRESHOLDS = {
    "excellent": 0.90,   # >90% → nearly perfect
    "good":      0.75,   # >75% → minor deviations
    "warning":   0.50,   # <50% → significant differences
    "bad":       0.30,   # <30% → poor overlap, needs review
    "critical":  0.10    # <10% → almost no overlap
}

# --------------------------
# Surface Distance Metrics
# --------------------------

# Chamfer Distance (mean squared ℓ₂, normalized by diag²)
# References:
#  • Fan et al., “A point set generation network for 3D object reconstruction from a single image,” CVPR 2017
#  • Mandikal et al., “3D‐LMNet: Latent embedding matching for accurate and diverse 3D point cloud reconstruction,” BMVC 2018
CHAMFER_THRESHOLDS = {
    "excellent": 0.02,   # <2%² → top‐tier
    "good":      0.05,   # <5%² → acceptable
    "warning":   0.10,   # <10%² → noticeable
    "bad":       0.20,   # <20%² → poor surface match
    "critical":  0.30    # <30%² → major reconstruction failures
}

# Hausdorff Distance (max ℓ₂, normalized by diag)
# References:
#  • Rusinkiewicz & Levoy, “Efficient variants of the ICP algorithm,” I3D 2001
#  • Gelfand et al., “Robust global registration,” SGP 2005
HAUSDORFF_THRESHOLDS = {
    "excellent": 0.01,   # <1% → no extreme outliers
    "good":      0.05,   # <5% → small localized errors
    "warning":   0.10,   # <10% → moderate outliers
    "bad":       0.20,   # <20% → large outliers present
    "critical":  0.30    # <30% → extreme surface mismatches
}

# --------------------------
# Geometric Feature Metrics
# --------------------------

# Normal Consistency (mean cosine similarity)
# References:
#  • Tulsiani et al., “Multi‐view consistency as supervisory signal for learning shape and pose prediction,” CVPR 2018
#  • Sinha et al., “Learning 3D shape completion under weak supervision,” ICCV 2017
NORMAL_CONSISTENCY_THRESHOLDS = {
    "excellent": 0.95,   # >0.95 → nearly identical normals
    "good":      0.85,   # >0.85 → generally consistent
    "warning":   0.70,   # <0.70 → significant distortion
    "bad":       0.50,   # <0.50 → poor normal agreement
    "critical":  0.30    # <0.30 → almost random normals
}

# Mean Curvature Error (dimensionless PCA proxy)
# References:
#  • Pauly et al., “Point‐based multiscale surface representation,” SIGGRAPH 2002
#  • Tang & Medioni, “Robust estimation of curvature and principal directions,” ICCV 1999
MEAN_CURVATURE_THRESHOLDS = {
    "excellent": 0.005,  # <0.005 → very tight curvature match
    "good":      0.010,  # <0.01 → minor variations
    "warning":   0.020,  # <0.02 → noticeable mismatches
    "bad":       0.040,  # <0.04 → poor curvature agreement
    "critical":  0.080   # <0.08 → extreme curvature failures
}

# --------------------------
# Point Distribution Metrics
# --------------------------

# Earth Mover’s Distance (mean ℓ₂, normalized by diag)
# References:
#  • Fan et al., “A point set generation network for 3D object reconstruction from a single image,” CVPR 2017
#  • Zhang et al., “DeepEMD: Few‐shot point cloud classification using Earth Mover’s Distance as a training loss,” CVPR 2020
EMD_THRESHOLDS = {
    "excellent": 0.01,   # <1% → exceptional match
    "good":      0.05,   # <5% → solid output
    "warning":   0.10,   # <10% → noticeable shift
    "bad":       0.20,   # <20% → poor distribution match
    "critical":  0.30    # <30% → extreme mismatch
}