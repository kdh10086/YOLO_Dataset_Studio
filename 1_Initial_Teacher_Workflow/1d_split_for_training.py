import os
import glob
import random
import shutil
import yaml
import argparse

# ==============================================================================
# Initiation Settings
# ==============================================================================
# This section defines global variables for script-specific default values.
# These variables act as fallbacks if a value is not specified via command-line
# arguments, allowing the script to run with predefined settings without
# needing constant modification of the main logic or config file.

# Default relative path to the dataset directory for splitting.
# This is primarily used for direct script execution or debugging.
# The main workflow relies on paths from _config.yaml or command-line arguments.
INIT_DATASET_DIR = None
# ==============================================================================

def split_and_organize_files(dataset_dir, train_ratio, image_formats):
    """
    Splits image and label files within a dataset folder into train/val sets.
    If already split, it resets and re-splits the files.
    """
    print("\n[STEP 1] Starting dataset split...")
    images_dir, labels_dir = os.path.join(dataset_dir, 'images'), os.path.join(dataset_dir, 'labels')

    # If train/val folders already exist, move all files back to the parent folder to reset
    for sub_dir_name in ['train', 'val']:
        for content_type_dir in [images_dir, labels_dir]:
            sub_dir_path = os.path.join(content_type_dir, sub_dir_name)
            if os.path.isdir(sub_dir_path):
                print(f"  - Resetting existing folder: '{os.path.basename(content_type_dir)}/{sub_dir_name}'")
                for filename in os.listdir(sub_dir_path):
                    shutil.move(os.path.join(sub_dir_path, filename), content_type_dir)
                os.rmdir(sub_dir_path)

    # Define and create the destination folders
    train_img_dir, val_img_dir = os.path.join(images_dir, 'train'), os.path.join(images_dir, 'val')
    train_lbl_dir, val_lbl_dir = os.path.join(labels_dir, 'train'), os.path.join(labels_dir, 'val')
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Get a list of image paths, sort, and then shuffle randomly
    image_paths = [p for fmt in image_formats for p in glob.glob(os.path.join(images_dir, f'*.{fmt}'))]
    if not image_paths:
        print(f"Error: No images to split in '{images_dir}'."); return False
        
    random.shuffle(image_paths)
    split_point = int(len(image_paths) * train_ratio)
    train_files, val_files = image_paths[:split_point], image_paths[split_point:]
    
    def move_files(file_list, dest_img_dir, dest_lbl_dir):
        """Helper function to move image-label pairs."""
        count = 0
        for img_path in file_list:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            if os.path.exists(label_path):
                shutil.move(img_path, os.path.join(dest_img_dir, img_name))
                shutil.move(label_path, os.path.join(dest_lbl_dir, label_name))
                count += 1
            else:
                print(f"  - Warning: Label file for '{img_name}' not found. Skipping.")
        return count
    
    labeled_train_count = move_files(train_files, train_img_dir, train_lbl_dir)
    labeled_val_count = move_files(val_files, val_img_dir, val_lbl_dir)

    print("Dataset split complete!")
    print(f"  - Training set: {labeled_train_count} files | Validation set: {labeled_val_count} files")
    return True

def generate_data_yaml(dataset_dir, classes_dict):
    """
    Generates the data.yaml file required for YOLO training.
    """
    print("\n[STEP 2] Generating data.yaml file...")
    # Use absolute path for the dataset to ensure it works in any environment
    dataset_abs_path = os.path.abspath(dataset_dir)
    class_names = [name for _, name in sorted(classes_dict.items())]
    yaml_content = {'path': dataset_abs_path, 'train': 'images/train', 'val': 'images/val', 'names': class_names}
    yaml_file_path = os.path.join(dataset_dir, 'data.yaml')
    try:
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
        print(f"data.yaml successfully created at: {yaml_file_path}")
        print("--- YAML Content ---\n" + yaml.dump(yaml_content, allow_unicode=True, sort_keys=False).strip() + "\n-------------------")
    except Exception as e:
        print(f"Error: Failed to generate data.yaml: {e}")

def main(config, args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_dir_rel = args.dataset if args.dataset is not None else INIT_DATASET_DIR if INIT_DATASET_DIR is not None else config['dataset_paths']['initial_labeled_dataset']
    dataset_dir = os.path.join(project_root, dataset_dir_rel)
    train_ratio = config['workflow_parameters']['train_split_ratio']
    image_formats = config['workflow_parameters']['image_format'].split(',')
    classes_dict = config['model_configurations']['classes']

    print("\n" + "="*50); print("Starting Dataset Split and YAML Generation."); print("="*50)
    print(f"  - Target Dataset: {dataset_dir}"); print(f"  - Train Split Ratio: {train_ratio}"); print("="*50)

    if not os.path.isdir(os.path.join(dataset_dir, 'images')):
        print(f"Error: 'images' directory not found in the target folder. Check the path: {dataset_dir}"); return
    if split_and_organize_files(dataset_dir, train_ratio, image_formats):
        generate_data_yaml(dataset_dir, classes_dict)
        print("\nAll tasks completed successfully.")
    else:
        print("\nYAML file was not created because the split operation failed.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f: config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: _config.yaml not found. Check if the file exists in the project root."); exit()

    parser = argparse.ArgumentParser(description="Splits a dataset into train/val and generates data.yaml.")
    parser.add_argument('--dataset', type=str, default=None, help="Relative path to the dataset to be split.")
    args = parser.parse_args()
    main(config, args)