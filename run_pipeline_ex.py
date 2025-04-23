import subprocess
import os
import time

# === CONFIGURATION ===
THREESTUDIO_IMAGE = "threestudio"
CONTAINER_NAME = "threestudio"
GEN_SCRIPT = "/home/dreamer/threestudio/scripts/gen_mesh.py"

# Paths on host (these should match actual mounted volumes)
shared_dir = os.path.abspath("../shared")  # adjust if needed
threestudio_dir = os.path.abspath("../threestudio")
mesh_filename = "generated_mesh.obj"
output_mesh_path = os.path.join(shared_dir, mesh_filename)

prompt = "a futuristic coffee mug"


def run_threestudio(prompt, output_path):
    print("Starting threestudio container...")
    subprocess.run([
        "docker", "run", "-dit", "--rm", "--gpus", "all",
        "--name", CONTAINER_NAME,
        "-v", f"{threestudio_dir}:/home/dreamer/threestudio",
        "-v", f"{shared_dir}:/shared",
        THREESTUDIO_IMAGE,
        "bash"
    ], check=True)

    print("Running 3D generation inside threestudio...")
    subprocess.run([
        "docker", "exec", CONTAINER_NAME,
        "python", GEN_SCRIPT,
        "--prompt", prompt,
        "--out", f"/shared/{mesh_filename}"
    ], check=True)

    print("Cleaning up threestudio container...")
    subprocess.run(["docker", "stop", CONTAINER_NAME], check=True)


def wait_for_file(path, timeout=300):
    print(f"Waiting for output file: {path}")
    start_time = time.time()
    while not os.path.exists(path):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"❌ Timeout waiting for {path}")
        time.sleep(1)
    print("Output file found.")


def run_pose_estimation(mesh_path):
    print("Running pose estimation...")
    # Replace with actual FoundationPose logic
    output_path = os.path.join(shared_dir, "pose_output.json")
    with open(output_path, "w") as f:
        f.write("{\"mock\": \"pose estimation result\"}")
    print(f"Pose estimation output saved to {output_path}")


# === PIPELINE ENTRY ===
if __name__ == "__main__":
    run_threestudio(prompt, output_mesh_path)
    wait_for_file(output_mesh_path)
    run_pose_estimation(output_mesh_path)
