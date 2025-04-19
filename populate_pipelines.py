import os
import shutil
import zipfile

# === CONFIG (HOTSDATA) ===
FILE_ID_HOTS = "199xCqJ11H6k5i2WohqzZNfOGbfhxv0Rk"
ZIP_PATH_HOTS = "hots_data.zip"
EXTRACT_TO_HOTS = "Pipelines/Process_HOTS"
TEMP_DIR_HOTS = "temp_folder"

# === CONFIG (FoundationPose ZIPS) ===
FOUNDATIONPOSE_ZIPS = {
    "1R9aurL8d1eUlXKXHTK-Zr0F2yDeRaVCa": "foundation_zip1.zip",
    "1wybpg0AZGyxEqzTXPn2vUYjMwjGhvJ8a": "foundation_zip2.zip"
}
EXTRACT_TO_FOUNDATIONPOSE = "FoundationPose"
TEMP_DIR_FOUNDATIONPOSE = "temp_foundation"

# === HOTS: Download & extract ===
print("[1] Downloading HOTS zip from Google Drive...")
os.system(f"gdown --id {FILE_ID_HOTS} -O {ZIP_PATH_HOTS}")

if not os.path.exists(ZIP_PATH_HOTS):
    print(f"ZIP file {ZIP_PATH_HOTS} was not downloaded correctly.")
    exit(1)

print(f"[2] Extracting {ZIP_PATH_HOTS} to temporary directory...")
os.makedirs(TEMP_DIR_HOTS, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH_HOTS, 'r') as zip_ref:
    zip_ref.extractall(TEMP_DIR_HOTS)

print(f"[3] Moving contents to {EXTRACT_TO_HOTS}...")
os.makedirs(EXTRACT_TO_HOTS, exist_ok=True)
for item in os.listdir(TEMP_DIR_HOTS):
    s = os.path.join(TEMP_DIR_HOTS, item)
    d = os.path.join(EXTRACT_TO_HOTS, item)
    if os.path.isdir(s):
        shutil.move(s, d)
    else:
        shutil.move(s, d)

# === HOTS: Cleanup ===
print("[4] Cleaning up HOTS zip and temp folder...")
os.remove(ZIP_PATH_HOTS)
shutil.rmtree(TEMP_DIR_HOTS)

# === FoundationPose: Download & extract ===
print("\n[5] Downloading FoundationPose zips...")
os.makedirs(EXTRACT_TO_FOUNDATIONPOSE, exist_ok=True)
os.makedirs(TEMP_DIR_FOUNDATIONPOSE, exist_ok=True)

for file_id, zip_name in FOUNDATIONPOSE_ZIPS.items():
    print(f"  ↳ Downloading {zip_name}...")
    os.system(f"gdown --id {file_id} -O {zip_name}")

    if not os.path.exists(zip_name):
        print(f"ZIP file {zip_name} was not downloaded correctly.")
        continue

    print(f"  ↳ Extracting {zip_name}...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR_FOUNDATIONPOSE)

    print(f"  ↳ Moving contents to {EXTRACT_TO_FOUNDATIONPOSE}...")
    for item in os.listdir(TEMP_DIR_FOUNDATIONPOSE):
        s = os.path.join(TEMP_DIR_FOUNDATIONPOSE, item)
        d = os.path.join(EXTRACT_TO_FOUNDATIONPOSE, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.move(s, d)
        else:
            shutil.move(s, d)

    os.remove(zip_name)
    shutil.rmtree(TEMP_DIR_FOUNDATIONPOSE)
    os.makedirs(TEMP_DIR_FOUNDATIONPOSE, exist_ok=True)

# === Final cleanup ===
shutil.rmtree(TEMP_DIR_FOUNDATIONPOSE)
print("\nAll done! Your data is in:")
print(f"  - {EXTRACT_TO_HOTS}/")
print(f"  - {EXTRACT_TO_FOUNDATIONPOSE}/")
