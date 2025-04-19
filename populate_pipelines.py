import os
import shutil
import zipfile

# === CONFIG 1: HOTS Data ===
FILE_ID_HOTS = "199xCqJ11H6k5i2WohqzZNfOGbfhxv0Rk"
ZIP_PATH_HOTS = "hots_data.zip"
EXTRACT_TO_HOTS = "Pipelines/Process_HOTS"
TEMP_DIR_HOTS = "temp_folder"

# === CONFIG 2: FoundationPose Zips ===
FOUNDATIONPOSE_ZIPS = {
    "1R9aurL8d1eUlXKXHTK-Zr0F2yDeRaVCa": "foundation_zip1.zip",
    "1wybpg0AZGyxEqzTXPn2vUYjMwjGhvJ8a": "foundation_zip2.zip"
}
EXTRACT_TO_FOUNDATIONPOSE = "FoundationPose"
TEMP_DIR_FOUNDATIONPOSE = "temp_foundation"

# === CONFIG 3: Linemod_3D_noise to Two Places ===
LINEMOD_ZIP_ID = "1kAiDcBYuOt5eFyyU7g-ixf_sLmXTkEAH"
LINEMOD_ZIP_NAME = "linemod_data.zip"
LINEMOD_TARGETS = [
    "Pipelines/Linemod_3D_noise",
    "Linemod_results/3d_genAI"
]
TEMP_DIR_LINEMOD = "temp_linemod"

# === Step 1: HOTS ===
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
    shutil.move(s, d)

print("[4] Cleaning up HOTS...")
os.remove(ZIP_PATH_HOTS)
shutil.rmtree(TEMP_DIR_HOTS)

# === Step 2: FoundationPose ===
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

shutil.rmtree(TEMP_DIR_FOUNDATIONPOSE)

# === Step 3: Linemod to Two Locations ===
print("\n[6] Downloading Linemod 3D GenAI zip...")
os.system(f"gdown --id {LINEMOD_ZIP_ID} -O {LINEMOD_ZIP_NAME}")

if not os.path.exists(LINEMOD_ZIP_NAME):
    print(f"ZIP file {LINEMOD_ZIP_NAME} was not downloaded correctly.")
    exit(1)

print(f"[7] Extracting {LINEMOD_ZIP_NAME} to temporary directory...")
os.makedirs(TEMP_DIR_LINEMOD, exist_ok=True)
with zipfile.ZipFile(LINEMOD_ZIP_NAME, 'r') as zip_ref:
    zip_ref.extractall(TEMP_DIR_LINEMOD)

print(f"[8] Copying to multiple destinations...")
for target in LINEMOD_TARGETS:
    os.makedirs(target, exist_ok=True)
    for item in os.listdir(TEMP_DIR_LINEMOD):
        s = os.path.join(TEMP_DIR_LINEMOD, item)
        d = os.path.join(target, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

print("[9] Cleaning up Linemod...")
os.remove(LINEMOD_ZIP_NAME)
shutil.rmtree(TEMP_DIR_LINEMOD)

# === Done ===
print("\nAll done! Your data has been extracted to:")
print(f"  - {EXTRACT_TO_HOTS}/")
print(f"  - {EXTRACT_TO_FOUNDATIONPOSE}/")
print(f"  - Pipelines/Linemod_3D_noise/")
print(f"  - Linemod_results/3d_genAI/")
