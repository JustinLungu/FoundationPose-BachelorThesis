import yaml

class GroundTruthFormatter:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
    
    def load_data(self):
        with open(self.input_file, "r") as file:
            return yaml.safe_load(file)
    
    def reformat(self):
        gt_data = self.load_data()
        formatted_data = {}
        for frame, objects in gt_data.items():
            frame_int = int(frame)
            obj_entry = objects[0]  # Assumes one object per frame
            # Reformat rotation into 3x3 and adjust translation units
            rotation = [obj_entry["cam_R_m2c"][i:i+3] for i in range(0, 9, 3)]
            translation = [t / 1000.0 for t in obj_entry["cam_t_m2c"]]
            obj_id = obj_entry["obj_id"]
            # Append the scale row (0,0,0,1)
            scale_matrix = [[0.0, 0.0, 0.0, 1.0]]
            transformation_matrix = [r + [t] for r, t in zip(rotation, translation)]
            transformation_matrix.extend(scale_matrix)
            if obj_id not in formatted_data:
                formatted_data[obj_id] = {}
            formatted_data[obj_id][frame_int] = {obj_id: transformation_matrix}
        return formatted_data
    
    def save(self, data):
        with open(self.output_file, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    
    def format_and_save(self):
        data = self.reformat()
        self.save(data)
        print(f"Fully reformatted YAML file saved as: {self.output_file}")

if __name__ == "__main__":
    formatter = GroundTruthFormatter("gt.yml", "gt_reformatted.yml")
    formatter.format_and_save()


import yaml

class ResultsFormatter:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
    
    def load_data(self):
        with open(self.input_file, "r") as file:
            return yaml.safe_load(file)
    
    def reformat(self, data):
        # Assumes the structure has a top-level key with frames to be renamed sequentially.
        key = next(iter(data))
        if isinstance(data[key], dict):
            frames = data[key]
            new_frames = {i: frames[k] for i, k in enumerate(sorted(frames.keys(), key=lambda x: int(x)))}
            data[key] = new_frames
        return data
    
    def save(self, data):
        with open(self.output_file, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    
    def format_and_save(self):
        data = self.load_data()
        data = self.reformat(data)
        self.save(data)
        print(f"Renamed YAML file saved as: {self.output_file}")

if __name__ == "__main__":
    formatter = ResultsFormatter("linemod_res.yml", "res_reformatted.yml")
    formatter.format_and_save()



