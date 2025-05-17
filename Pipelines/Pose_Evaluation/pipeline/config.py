# Input data structure
LINEMOD_ROOT = "linemod_results"
OBJECT_IDS = [1, 4, 6, 9, 10]  # Objects to evaluate
POSE_METHODS = ["original", "3d_genAI", "normal", "gaussian", "outlier", "speckle"]  # Methods to compare

# Frame selection (can be made dynamic per object if needed)
FRAME_IDX = 0  

# Zoom factors (remain the same)
ZOOMED_ZOOM_FACTOR = 0.3
FULL_ZOOM_FACTOR = 0.4
GIF_ZOOM_FACTOR = 0.5
ROTATION_ANGLES = [90, -90, 0]

# Outlier thresholds and labels (remain the same)
OUTLIER_THRESHOLDS = (10, 0.05, 0.1, 0.05)
TREND_THRESHOLDS = {
    "rotation": [5, 10],
    "translation": [0.01, 0.05],
    "pose": [0.1, 0.3],
    "add": [0.01, 0.05],
}
LABELS = [
    ("Rotation Error", "Degrees", "blue", "rotation"),
    ("Translation Error", "Meters", "orange", "translation"),
    ("Pose Error", "Error", "green", "pose"),
    ("ADD Error", "Meters", "purple", "add"),
]