import os
import glob
import shutil
import yaml
import argparse
from tqdm import tqdm

# ==============================================================================
# Initiation Settings
# ==============================================================================
# These global variables define default parameters for the dataset merge tool.
# They can be overridden by command-line arguments or settings in _config.yaml.

# Default list of relative paths to the input dataset directories to be merged.
INIT_INPUT_DIRS = None

# Default relative path for the new, merged output dataset.
INIT_OUTPUT_DIR = None

# Default setting for overwriting the output directory if it already exists.
# If False, the script will stop to prevent accidental data loss.
INIT_EXIST_OK = False
# ==============================================================================

def find_all_image_paths(dataset_dir):
    """Finds all image paths in a dataset, auto-detecting its structure."""
    image_paths, image_formats = [], ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    is_split = os.path.isdir(os.path.join(dataset_dir, 'images', 'train'))
    print(f"  - Analyzing dataset '{dataset_name}'... (Structure: {'Split' if is_split else 'Unified'})")
    base_images_dir = os.path.join(dataset_dir, 'images')
    dirs_to_scan = [os.path.join(base_images_dir, sub) for sub in ['train','val']] if is_split else [base_images_dir]
    for d in dirs_to_scan:
        for fmt in image_formats: image_paths.extend(glob.glob(os.path.join(d, fmt)))
    return sorted(image_paths)

def merge_and_rename(all_image_paths, output_dir, exist_ok=False):
    """Merges all given image and label files into a single folder and renames them sequentially."""
    if os.path.exists(output_dir):
        if exist_ok:
            shutil.rmtree(output_dir)
            print(f"\nWarning: Deleting existing output folder '{output_dir}' as --exist_ok is set.")
        else:
            print(f"\nError: Output directory '{output_dir}' already exists.")
            print("To overwrite, run the script again with the --exist_ok flag.")
            return

    output_images_dir = os.path.join(output_dir, 'images'); os.makedirs(output_images_dir)
    output_labels_dir = os.path.join(output_dir, 'labels'); os.makedirs(output_labels_dir)
    print(f"\nMerging and renaming {len(all_image_paths)} files...")
    for i, img_path in enumerate(tqdm(all_image_paths, desc="Merging files")):
        new_basename, ext = f"{i:06d}", os.path.splitext(img_path)[1]
        new_image_filename, new_label_filename = f"{new_basename}{ext}", f"{new_basename}.txt"
        label_path = os.path.splitext(img_path.replace('images','labels',1))[0] + '.txt'
        shutil.copy2(img_path, os.path.join(output_images_dir, new_image_filename))
        if os.path.exists(label_path): shutil.copy2(label_path, os.path.join(output_labels_dir, new_label_filename))
    print("\nMerge and rename complete!"); print(f"  - A total of {len(all_image_paths)} files were saved to '{output_dir}'.")

def main(config, args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    input_dirs_rel = args.inputs or INIT_INPUT_DIRS or config['dataset_paths']['merge_tool_inputs']
    output_dir_rel = args.output or INIT_OUTPUT_DIR or config['dataset_paths']['merged_dataset_for_retrain']
    exist_ok = args.exist_ok or (INIT_EXIST_OK is True)

    if not input_dirs_rel or not output_dir_rel:
        print("Error: Input or output datasets not specified."); return
    
    print("\n" + "="*50); print("Starting Dataset Merge."); print("="*50)
    print("Input Datasets:" + "".join([f"\n  - {p}" for p in input_dirs_rel]))
    print(f"\nOutput Dataset:\n  - {output_dir_rel}"); print("="*50)

    all_image_paths = []
    for rel_path in input_dirs_rel:
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.isdir(abs_path): print(f"Error: Input path not found -> {abs_path}"); return
        image_paths = find_all_image_paths(abs_path)
        print(f"    -> Found {len(image_paths)} images in '{rel_path}'"); all_image_paths.extend(image_paths)
    
    merge_and_rename(all_image_paths, os.path.join(project_root, output_dir_rel), exist_ok)
    print("\nAll tasks completed successfully.")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f: config = yaml.safe_load(f)
    except FileNotFoundError: print("Error: _config.yaml not found."); exit()
    parser = argparse.ArgumentParser(description="Merges multiple datasets into a new single dataset.")
    parser.add_argument('--inputs', nargs='+', default=None, help="List of relative paths for the datasets to be merged.")
    parser.add_argument('--output', type=str, default=None, help="Relative path for the final created dataset.")
    parser.add_argument('--exist_ok', action='store_true', help="If set, overwrites the output directory if it exists.")
    args = parser.parse_args()
    main(config, args)