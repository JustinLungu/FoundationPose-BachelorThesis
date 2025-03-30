import yaml

# Load the ground truth YAML file
input_file = "gt.yml"
output_file = "gt_reformatted.yml"

with open(input_file, "r") as file:
    gt_data = yaml.safe_load(file)

# Reformatted structure to match linemod_res.yml
formatted_data = {}

for frame, objects in gt_data.items():
    # Ensure frame is an integer
    frame_int = int(frame)
    
    # Extract object details
    obj_entry = objects[0]  # Assuming one object per frame
    rotation = [obj_entry["cam_R_m2c"][i:i+3] for i in range(0, 9, 3)]  # Convert to 3x3
    translation = [t / 1000.0 for t in obj_entry["cam_t_m2c"]]  # Convert from mm to m
    obj_id = obj_entry["obj_id"]
    
    # Add the missing scale transformation (0,0,0,1)
    scale_matrix = [[0.0, 0.0, 0.0, 1.0]]
    
    # Reformat rotation and translation into the required 4x4 format
    transformation_matrix = [r + [t] for r, t in zip(rotation, translation)]
    transformation_matrix.extend(scale_matrix)  # Append scale row
    
    # Structure to match linemod_res.yml
    if obj_id not in formatted_data:
        formatted_data[obj_id] = {}
    
    formatted_data[obj_id][frame_int] = {obj_id: transformation_matrix}

# Save the fully formatted YAML file
with open(output_file, "w") as file:
    yaml.dump(formatted_data, file, default_flow_style=False, sort_keys=False)

print(f"Fully reformatted YAML file saved as: {output_file}")