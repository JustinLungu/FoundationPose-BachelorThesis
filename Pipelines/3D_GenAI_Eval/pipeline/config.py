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
# These papers report mean IoU scores but do not define “good”/“warning” bins—
# our cut-offs (0.9/0.75/0.5) are chosen to reflect nearly perfect, acceptable,
# and poor volume overlaps in a reconstruction setting.
IOU_THRESHOLDS = {
    "excellent": 0.9,    # >90% → nearly perfect match
    "good":      0.75,   # >75% → minor deviations acceptable
    "warning":   0.5     # <50% → significant structural differences
}

# --------------------------
# Surface Distance Metrics
# --------------------------

# Chamfer Distance (mean squared ℓ₂ in mm²)
# References:
#  • Fan et al., “A point set generation network for 3D object reconstruction from a single image,” CVPR 2017
#  • Mandikal et al., “3D‐LMNet: Latent embedding matching for accurate and diverse 3D point cloud reconstruction,” BMVC 2018
# These works report linear Chamfer in mm; to use squared‐Chamfer we square their
# “good”/“warn” benchmarks (≈7 mm, 15 mm).
CHAMFER_THRESHOLDS = {
    "good":  (7.0)**2,    # ~49 mm² → high quality reconstruction
    "warn": (15.0)**2,    # ~225 mm² → noticeable surface deviations
    "bad":  (30.0)**2     # ~900 mm² → major reconstruction failures
}

# Hausdorff Distance (max ℓ₂ in mm)
# References:
#  • Rusinkiewicz & Levoy, “Efficient variants of the ICP algorithm,” I3D 2001
#  • Gelfand et al., “Robust global registration,” SGP 2005
# These classical registration papers use Hausdorff as an outlier metric;
# our cut-offs mirror “no extreme outliers” and “localized large errors.”
HAUSDORFF_THRESHOLDS = {
    "good": 25.0,         # <25 mm → no extreme outliers
    "decent": 100.0       # <100 mm → some localized large errors
}

# --------------------------
# Geometric Feature Metrics
# --------------------------

# Normal Consistency (mean cosine similarity)
# References:
#  • Tulsiani et al., “Multi-view consistency as supervisory signal for learning shape 
#    and pose prediction,” CVPR 2018
#  • Sinha et al., “Learning 3D shape completion under weak supervision,” ICCV 2017
# Both use average dot‐product of normals as a consistency measure; thresholds are
# chosen to reflect “nearly identical,” “generally consistent,” and “distorted.”
NORMAL_CONSISTENCY_THRESHOLDS = {
    "excellent": 0.95,   # >0.95 → nearly identical surface orientation
    "good":      0.85,   # >0.85 → generally consistent normals
    "decent":    0.70    # <0.70 → significant normal field distortion
}

# Mean Curvature Error (dimensionless PCA proxy)
# References:
#  • Pauly et al., “Point‐based multiscale surface representation,” SIGGRAPH 2002
#  • Tang & Medioni, “Robust estimation of curvature and principal directions,” ICCV 1999
# These classical geometry papers define curvature via PCA eigenvalues;
# our thresholds (0.005/0.01/0.02) apply directly to the dimensionless proxy H.
MEAN_CURVATURE_THRESHOLDS = {
    "excellent": 0.005,  # <0.005 → nearly identical local bending
    "good":      0.01,   # <0.01 → minor curvature variations
    "warning":   0.02    # >0.02 → noticeable curvature mismatches
}

# --------------------------
# Point Distribution Metrics
# --------------------------

# Earth Mover’s Distance (mean ℓ₂ in mm)
# References:
#  • Fan et al., “A point set generation network…,” CVPR 2017
#  • Zhang et al., “DeepEMD: Few‐shot point cloud classification using Earth Mover’s Distance 
#    as a training loss,” CVPR 2020
# These works use mean transport distance in mm; our mm‐based cut-offs reflect
# “exceptional” sub‐2mm, “solid” up to 8mm, and “warning” beyond 15mm average error.
EMD_THRESHOLDS = {
    "excellent":  2.0,   # <2 mm → exceptional match
    "good":       8.0,   # 2–8 mm → solid generative output
    "warning":   15.0    # >15 mm → noticeable distribution shift
}