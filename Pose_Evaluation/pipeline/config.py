# Input data paths
GT_YAML = "reformatted/gt_reformatted.yml"
PRED_YAML = "reformatted/res_reformatted.yml"
PLY_PATH = "data/obj_01.ply"

# Frame selection
FRAME_IDX = 0

# Zoom factors
ZOOMED_ZOOM_FACTOR = 0.3
FULL_ZOOM_FACTOR = 0.4
GIF_ZOOM_FACTOR = 0.5

ROTATION_ANGLES = [90, -90, 0]

# Output paths
ZOOMED_IMG_PATH = f"plots/frame_{FRAME_IDX}_zoomed.png"
FULL_IMG_PATH = f"plots/frame_{FRAME_IDX}_full.png"
ANNOTATED_IMG_PATH = f"plots/frame_{FRAME_IDX}_annotated.png"
GIF_PATH = "plots/orbit_animation.gif"
OUTLIER_PLOT_PATH = "plots/error_outliers.png"
TREND_PLOT_PATH = "plots/error_trends.png"
DISTRIBUTION_PLOT_PATH = "plots/error_distributions.png"

# Outlier thresholds (rotation, translation, pose, add)
OUTLIER_THRESHOLDS = (10, 0.05, 0.1, 0.05)

# Trend/Histogram thresholds
TREND_THRESHOLDS = {
    "rotation": [5, 10],
    "translation": [0.01, 0.05],
    "pose": [0.1, 0.3],
    "add": [0.01, 0.05],
}

# Labels and colors
LABELS = [
    ("Rotation Error", "Degrees", "blue", "rotation"),
    ("Translation Error", "Meters", "orange", "translation"),
    ("Pose Error", "Error", "green", "pose"),
    ("ADD Error", "Meters", "purple", "add"),
]