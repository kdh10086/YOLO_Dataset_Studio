import os
import glob
import shutil
import random
import yaml
import argparse
from tqdm import tqdm

# ==============================================================================
# Initiation Settings
# ==============================================================================
# These global variables define default parameters for the random sampling tool.
# They can be overridden by command-line arguments or settings in _config.yaml.

# Default relative path to the source dataset directory to sample from.
INIT_SOURCE_DIR = None

# Default relative path for the new, smaller sampled dataset.
INIT_OUTPUT_DIR = None

# Default ratio of the dataset to sample (e.g., 0.1 for 10%).
INIT_SAMPLE_RATIO = None

# Default setting for overwriting the output directory if it already exists.
# If False, the script will stop to prevent accidental data loss.
INIT_EXIST_OK = False
# ==============================================================================

def main(config, args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    source_dir_rel = args.source or INIT_SOURCE_DIR or config['dataset_paths']['sampling_tool_source']
    source_dir = os.path.join(project_root, source_dir_rel)
    output_dir_rel = args.output or INIT_OUTPUT_DIR or config['dataset_paths']['sampling_tool_output']
    output_dir = os.path.join(project_root, output_dir_rel)
    sample_ratio = args.ratio or INIT_SAMPLE_RATIO or config['workflow_parameters']['sampling_tool_ratio']
    exist_ok = args.exist_ok or (INIT_EXIST_OK is True)
    
    print("\n" + "="*50); print("Starting Random Dataset Sampling."); print("="*50)
    print(f"  - Source Dataset: {source_dir}"); print(f"  - Sample Save Path: {output_dir}")
    print(f"  - Sampling Ratio: {sample_ratio:.2%}"); print("="*50)
    
    if not os.path.isdir(source_dir): print(f"Error: Source path not found: {source_dir}"); return
    if not (0.0 < sample_ratio <= 1.0): print(f"Error: Sampling ratio must be between 0 and 1: {sample_ratio}"); return

    print("Scanning for all image files in the source dataset...")
    image_formats = config['workflow_parameters']['image_format'].split(',')
    all_image_paths = [p for fmt in image_formats for p in glob.glob(os.path.join(source_dir, '**', f'*.{fmt}'), recursive=True)]
    file_pairs = []
    for img_path in all_image_paths:
        label_path = os.path.splitext(img_path.replace(f'{os.path.sep}images{os.path.sep}', f'{os.path.sep}labels{os.path.sep}', 1))[0] + '.txt'
        if os.path.exists(label_path): file_pairs.append({'image': img_path, 'label': label_path})
    
    if not file_pairs: print("Error: No image-label pairs found in the source dataset."); return
    print(f"Found a total of {len(file_pairs)} image-label pairs.")
    
    random.shuffle(file_pairs)
    num_to_sample = int(len(file_pairs) * sample_ratio)
    sampled_files = file_pairs[:num_to_sample]
    print(f"Sampling {num_to_sample} of these pairs.")

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
    
    print("\nStarting to copy sample files...")
    for pair in tqdm(sampled_files, desc="Copying files"):
        shutil.copy2(pair['image'], os.path.join(output_images_dir, os.path.basename(pair['image'])))
        shutil.copy2(pair['label'], os.path.join(output_labels_dir, os.path.basename(pair['label'])))

    print("\nSampling complete!"); print(f"  - A total of {len(sampled_files)} files were saved to '{output_dir}'.")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f: config = yaml.safe_load(f)
    except FileNotFoundError: print("Error: _config.yaml not found."); exit()
    parser = argparse.ArgumentParser(description="Randomly samples a portion of a large dataset.")
    parser.add_argument('--source', type=str, default=None, help="Relative path to the source dataset to sample from.")
    parser.add_argument('--output', type=str, default=None, help="Relative path for the created sample dataset.")
    parser.add_argument('--ratio', type=float, default=None, help="Sampling ratio (0.0 to 1.0).")
    parser.add_argument('--exist_ok', action='store_true', help="If set, overwrites the output directory if it exists.")
    args = parser.parse_args()
    main(config, args)