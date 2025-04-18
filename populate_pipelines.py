import shutil
import os

# Step 1: Download using gdown (if running from within Python)
os.system(
    "gdown --folder https://drive.google.com/drive/folders/1T1o0DMqS8dH2-oluldsj2IADx5jCnd-b -O temp_folder"
)

# Step 2: Move to target location
src_dir = "temp_folder"
dst_dir = "Pipelines/Process_HOTS"

# Make sure destination exists
os.makedirs(dst_dir, exist_ok=True)

# Move contents
for item in os.listdir(src_dir):
    s = os.path.join(src_dir, item)
    d = os.path.join(dst_dir, item)
    if os.path.isdir(s):
        shutil.move(s, d)
    else:
        shutil.move(s, d)

shutil.rmtree("temp_folder")
