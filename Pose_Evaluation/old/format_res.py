import yaml

# Load the YAML file
input_file = "linemod_res.yml"
output_file = "res_reformatted.yml"

with open(input_file, "r") as file:
    data = yaml.safe_load(file)

# Rename frames to sequential integers
if isinstance(data, dict):
    key = next(iter(data))  # Get the first key (assumes the structure is known)
    if isinstance(data[key], dict):
        frames = data[key]
        new_frames = {i: frames[k] for i, k in enumerate(sorted(frames.keys(), key=lambda x: int(x)))}
        data[key] = new_frames

# Save the updated YAML file
with open(output_file, "w") as file:
    yaml.dump(data, file, default_flow_style=False, sort_keys=False)

print(f"Renamed YAML file saved as: {output_file}")
