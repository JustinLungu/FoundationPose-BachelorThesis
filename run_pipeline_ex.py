import subprocess
import os
import time

# Configurable paths
shared_dir = "/shared"
mesh_filename = "generated_mesh.obj"
output_mesh_path = os.path.join(shared_dir, mesh_filename)
prompt = "a futuristic coffee mug"

def run_threestudio(prompt, output_path):
    print("Starting threestudio container...")
    subprocess.run([
        "docker", "run", "-dit", "--rm", "--gpus", "all",
        "--name", "threestudio",
        "-v", "/workspace/threestudio:/home/dreamer/threestudio",
        "-v", "/shared:/shared",
        "threestudio", "bash"
    ], check=True)

    print("Running 3D generation...")
    subprocess.run([
        "docker", "exec", "threestudio",
        "python", "/home/dreamer/threestudio/scripts/gen_mesh.py",
        "--prompt", prompt,
        "--out", output_path
    ], check=True)

    print("Stopping threestudio container...")
    subprocess.run(["docker", "stop", "threestudio"], check=True)

def wait_for_file(path, timeout=60):
    print(f"⏳ Waiting for output file: {path}")
    start_time = time.time()
    while not os.path.exists(path):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for {path}")
        time.sleep(1)
    print("Output file found.")

def run_pose_estimation(mesh_path):
    print("Running pose estimation...")
    # Placeholder for actual FoundationPose logic
    # For example:
    # subprocess.run(["python", "run_foundationpose.py", "--mesh", mesh_path], check=True)
    output_path = os.path.join(shared_dir, "pose_output.json")
    with open(output_path, "w") as f:
        f.write("{\"mock\": \"pose estimation result\"}")
    print(f"Pose estimation output saved to {output_path}")

# === Main Pipeline ===
if __name__ == "__main__":
    run_threestudio(prompt, output_mesh_path)
    wait_for_file(output_mesh_path)
    run_pose_estimation(output_mesh_path)
