import os
import pandas as pd

class HOTSDirectoryCreator:
    def __init__(self, label_mapping_file, output_dir="HOTS_Processed"):
        self.label_mapping_file = label_mapping_file
        self.output_dir = output_dir
        self.id_to_name_mapping = {}

        # Load label mapping
        self._load_label_mapping()

        # Create base directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_label_mapping(self):
        """Loads the label mapping from the CSV file."""
        label_mapping_df = pd.read_csv(self.label_mapping_file)
        self.id_to_name_mapping = dict(zip(label_mapping_df["ID"], label_mapping_df["Instance"]))

    def create_structure(self):
        """Creates the directory structure for each object."""
        subfolders = ["RGB", "Depth", "Mask", "Mesh"]

        for object_name in self.id_to_name_mapping.values():
            object_dir = os.path.join(self.output_dir, object_name)
            os.makedirs(object_dir, exist_ok=True)

            for subfolder in subfolders:
                os.makedirs(os.path.join(object_dir, subfolder), exist_ok=True)

        print(f"Directory structure created inside '{self.output_dir}'.")

# Example Usage
if __name__ == "__main__":
    directory_creator = HOTSDirectoryCreator(label_mapping_file="HOTS_v1/label_mapping.csv")
    directory_creator.create_structure()
