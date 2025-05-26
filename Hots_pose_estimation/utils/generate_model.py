import subprocess
import shutil
import os
import stat


class ModelGenerator:
    def __init__(self, threestudio_dir="/app/threestudio/docker", script="run_container.sh"):
        self.threestudio_dir = threestudio_dir
        self.script = script

    def generate(self, object_name: str):
        prompt = object_name.replace("_", " ")
        print(f"[ModelGenerator] Generating 3D model for: '{prompt}'")

        abs_script_path = os.path.join(self.threestudio_dir, self.script)

        if not os.path.exists(abs_script_path):
            raise FileNotFoundError(f"Cannot find script at {abs_script_path}")

        try:
            subprocess.run(["bash", abs_script_path, prompt], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ModelGenerator] Error during generation: {e}")
            return

        print("[ModelGenerator] Generation completed.")
        self._demo_move_exported_directory(object_name)

    def _demo_move_exported_directory(self, object_name: str):
        """Move the most recent ThreeStudio export folder to the mesh directory for the given object"""

        # Search for dreamfusion-sd in all likely locations
        candidate_roots = ["/app", "/workspace", "/threestudio", "/home", "/tmp"]
        outputs_dir = None

        print("[ModelGenerator] Searching for 'dreamfusion-sd' folder...")
        for root in candidate_roots:
            for dirpath, dirnames, filenames in os.walk(root):
                if "dreamfusion-sd" in dirnames:
                    outputs_dir = os.path.join(dirpath, "dreamfusion-sd")
                    break
            if outputs_dir:
                break

        if not outputs_dir or not os.path.exists(outputs_dir):
            print("[ModelGenerator] ERROR: Could not find 'dreamfusion-sd' in any known location.")
            return

        print(f"[ModelGenerator] Using outputs directory: {outputs_dir}")

        destination_path = os.path.join("data", "HOTS_Processed_demo", object_name, "mesh")

        # Step 1: Find the trial folder that includes the object name
        trial_folder = None
        try:
            for folder in os.listdir(outputs_dir):
                if object_name in folder:
                    trial_folder = os.path.join(outputs_dir, folder)
                    break
        except Exception as e:
            print(f"[ModelGenerator] ERROR reading {outputs_dir}: {e}")
            return

        if not trial_folder or not os.path.exists(trial_folder):
            print(f"[ModelGenerator] Trial folder for '{object_name}' not found in {outputs_dir}")
            return

        # Step 2: Find export folder inside 'save'
        save_dir = os.path.join(trial_folder, "save")
        export_folder = None
        if os.path.exists(save_dir):
            for subfolder in os.listdir(save_dir):
                if "export" in subfolder:
                    export_folder = os.path.join(save_dir, subfolder)
                    break

        if not export_folder or not os.path.exists(export_folder):
            print(f"[ModelGenerator] Export folder not found in {save_dir}")
            return

        # Step 3: Ensure destination exists and move
        os.makedirs(destination_path, exist_ok=True)
        for item in os.listdir(export_folder):
            src = os.path.join(export_folder, item)
            dst = os.path.join(destination_path, item)
            shutil.move(src, dst)

        os.chmod(destination_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        print(f"[ModelGenerator] Exported mesh moved from {export_folder} to {destination_path}")