import os
import shutil
import zipfile

# === CONFIG ===
FILE_ID = "199xCqJ11H6k5i2WohqzZNfOGbfhxv0Rk"
ZIP_PATH = "hots_data.zip"
EXTRACT_TO = "Pipelines/Process_HOTS"
TEMP_DIR = "temp_folder"

# === 1. Download ZIP ===
print("[1] Downloading ZIP from Google Drive...")
os.system(f"gdown --id {FILE_ID} -O {ZIP_PATH}")

# === 2. Extract ZIP ===
if not os.path.exists(ZIP_PATH):
    print(f"ZIP file {ZIP_PATH} was not downloaded correctly.")
    exit(1)

print(f"[2] Extracting {ZIP_PATH} to temporary directory...")
os.makedirs(TEMP_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(TEMP_DIR)

# === 3. Move contents to target location ===
print(f"[3] Moving files to {EXTRACT_TO}...")
os.makedirs(EXTRACT_TO, exist_ok=True)

for item in os.listdir(TEMP_DIR):
    s = os.path.join(TEMP_DIR, item)
    d = os.path.join(EXTRACT_TO, item)
    if os.path.isdir(s):
        shutil.move(s, d)
    else:
        shutil.move(s, d)

# === 4. Clean up ===
print("[4] Cleaning up...")
os.remove(ZIP_PATH)
shutil.rmtree(TEMP_DIR)

print("All done! Your data is in Pipelines/Process_HOTS/")
