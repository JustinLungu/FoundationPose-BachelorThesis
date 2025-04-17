#!/usr/bin/env python3
import os
import sys

# Ensure PyYAML is available
try:
    import yaml
except ModuleNotFoundError:
    print("Error: PyYAML is not installed. Install it with 'pip install pyyaml' or 'conda install pyyaml'.")
    sys.exit(1)


def split_yaml(input_path: str, output_dir: str):
    """
    Split a YAML file into multiple files based on its top-level keys.

    Each top-level key will produce a separate YAML file named:
        {base_filename}_{key}.yml

    :param input_path: Path to the input YAML file
    :param output_dir: Directory where split files will be written
    """
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)

    base = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    for key, subtree in data.items():
        out_name = f"{base}_{key}.yml"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, 'w') as out_f:
            yaml.safe_dump({key: subtree}, out_f, sort_keys=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    # Automatically find and process every 'linemod_res.yml' under this directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    input_filename = "linemod_res.yml"

    # Collect all directories containing the target file
    matches = []
    for root, dirs, files in os.walk(script_dir):
        if input_filename in files:
            matches.append(root)

    if not matches:
        print(f"Error: '{input_filename}' not found under {script_dir}")
        sys.exit(1)

    # Split each found file in-place
    for folder in matches:
        path = os.path.join(folder, input_filename)
        print(f"Splitting {path} …")
        split_yaml(path, folder)
