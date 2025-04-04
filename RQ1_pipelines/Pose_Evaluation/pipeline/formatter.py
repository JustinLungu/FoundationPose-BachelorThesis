import yaml


class YAMLFormatter:
    def __init__(self):
        pass

    def reformat_predictions(self, input_file: str, output_file: str):
        """Reformats result YAMLs by renaming frames to sequential integers."""
        with open(input_file, "r") as file:
            data = yaml.safe_load(file)

        if isinstance(data, dict):
            key = next(iter(data))
            if isinstance(data[key], dict):
                frames = data[key]
                new_frames = {
                    i: frames[k] for i, k in enumerate(sorted(frames.keys(), key=lambda x: int(x)))
                }
                data[key] = new_frames

        with open(output_file, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)

        print(f"[✓] Reformatted prediction YAML saved as: {output_file}")

    def reformat_ground_truth(self, input_file: str, output_file: str):
        """Reformats ground truth YAMLs to match result file structure."""
        with open(input_file, "r") as file:
            gt_data = yaml.safe_load(file)

        formatted_data = {}

        for frame, objects in gt_data.items():
            frame_int = int(frame)
            obj_entry = objects[0]  # Assuming one object per frame
            rotation = [obj_entry["cam_R_m2c"][i:i+3] for i in range(0, 9, 3)]
            translation = [t / 1000.0 for t in obj_entry["cam_t_m2c"]]
            obj_id = obj_entry["obj_id"]

            # 4x4 transformation matrix
            transformation_matrix = [r + [t] for r, t in zip(rotation, translation)]
            transformation_matrix.append([0.0, 0.0, 0.0, 1.0])  # Homogeneous row

            if obj_id not in formatted_data:
                formatted_data[obj_id] = {}

            formatted_data[obj_id][frame_int] = {obj_id: transformation_matrix}

        with open(output_file, "w") as file:
            yaml.dump(formatted_data, file, default_flow_style=False, sort_keys=False)

        print(f"[✓] Reformatted GT YAML saved as: {output_file}")
