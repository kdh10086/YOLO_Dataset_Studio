import os
import sys
import glob
import yaml
from pathlib import Path
from datetime import datetime

# Import toolkit functions
from toolkit import data_handler, labeling, training

# --- Dependency & Environment Check ---
ROS2_ENABLED = False
GUI_ENABLED = False

def check_environment():
    """Checks for ROS2 and GUI capabilities, warning the user if features are disabled."""
    global ROS2_ENABLED, GUI_ENABLED
    
    # 1. Check for ROS2
    try:
        from rosbag2_py import SequentialReader
        from cv_bridge import CvBridge
        ROS2_ENABLED = True
    except ImportError:
        print("="*80)
        print(" WARNING: ROS2 Dependencies Not Found ".center(80, "="))
        print("="*80)
        print("The ROS2 bag extraction feature (Option 1) is disabled.")
        print("To enable it, please ensure you have a valid ROS2 installation (e.g., Humble)")
        print("and have sourced the setup script before running this toolkit.")
        print("Example: source /opt/ros/humble/setup.bash")
        print("="*80)

    # 2. Check for GUI Support
    if os.name == 'nt' or os.environ.get('DISPLAY'):
        GUI_ENABLED = True
    else:
        print("="*80)
        print(" WARNING: GUI Support Not Detected ".center(80, "="))
        print("="*80)
        print("GUI-dependent features (Labeling Tool, Interactive Extraction) are disabled.")
        print("To enable them, run this script in a graphical desktop environment.")
        print("="*80)

# --- UI & Input Helpers ---
def print_cancel_message():
    print("-> Tip: Enter 'c' at any prompt to cancel and return to the main menu.")

def get_input(prompt, default=None):
    """Gets user input with support for a default value and cancellation."""
    if default is not None and default != '':
        full_prompt = f"{prompt} [Default: {default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    user_input = input(full_prompt).strip()
    if user_input.lower() == 'c':
        return 'c'
    
    return user_input or default if default is not None else user_input

# --- Global State ---
registered_datasets = []
config = {}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_header():
    print("=" * 80)
    print(" ROS2 YOLO Toolkit - Integrated Environment ".center(80, "="))
    print("=" * 80)
    print()

def display_main_ui():
    clear_screen()
    display_header()
    
    menu = {
        "--- Data Preparation ---": {
            "1": "Extract Images from ROS Bag",
            "2": "Launch Integrated Labeling Tool",
            "3": "Split Dataset for Training",
        },
        "--- Training & Inference ---": {
            "4": "Train a Model (Teacher/Student)",
            "5": "Auto-label a Dataset with a Teacher",
        },
        "--- Utilities ---": {
            "6": "Merge Datasets",
            "7": "Random Sample from Dataset",
            "8": "Add New Dataset Directory",
        },
        "--- Exit ---": {
            "0": "Exit Toolkit",
        }
    }

    if not ROS2_ENABLED:
        menu["--- Data Preparation ---"]["1"] += " (Disabled: ROS2 Not Found)"
    if not GUI_ENABLED:
        if ROS2_ENABLED: # Avoid double-messaging if both are off
            menu["--- Data Preparation ---"]["1"] += " (Interactive Modes Disabled)"
        menu["--- Data Preparation ---"]["2"] += " (Disabled: GUI Not Available)"

    left_panel_lines = []
    for phase, options in menu.items():
        left_panel_lines.append(phase)
        for key, value in options.items():
            left_panel_lines.append(f"  [{key}] {value}")
        left_panel_lines.append("")

    right_panel_lines = ["--- Registered Datasets ---"]
    if not registered_datasets:
        right_panel_lines.append(" No datasets registered yet.")
        right_panel_lines.append(" Use option [8] to add one.")
    else:
        for i, path in enumerate(registered_datasets, 1):
            right_panel_lines.append(f"  [{i}] {path}")
    right_panel_lines.append("")

    max_lines = max(len(left_panel_lines), len(right_panel_lines))
    for i in range(max_lines):
        left_line = left_panel_lines[i] if i < len(left_panel_lines) else ""
        right_line = right_panel_lines[i] if i < len(right_panel_lines) else ""
        print(f"{left_line:<55}{right_line}")
    print("-" * 80)

def get_dataset_from_user(prompt="Select a dataset to use (by number)"):
    if not registered_datasets: 
        print("\n[Error] No datasets registered.")
        return None
    
    while True:
        choice_str = get_input(prompt)
        if choice_str == 'c': return 'c'
        try:
            choice_int = int(choice_str)
            if 1 <= choice_int <= len(registered_datasets):
                return registered_datasets[choice_int - 1]
            else:
                print(f"[Error] Invalid selection.")
        except (ValueError, IndexError):
            print("[Error] Invalid input. Please enter a number.")

def get_multiple_datasets_from_user():
    if not registered_datasets:
        print("\n[Error] No datasets registered.")
        return []
    
    selected_datasets = []
    while True:
        print("\n--- Select datasets to merge (enter 'c' to finish) ---")
        for i, path in enumerate(registered_datasets, 1):
            print(f"  [{i}] {path} {'(selected)' if path in selected_datasets else ''}")
        
        choice_str = get_input("Enter number to add/remove (or press Enter to finish)")
        if choice_str == 'c' or choice_str == '':
            break
        
        try:
            path = registered_datasets[int(choice_str) - 1]
            if path in selected_datasets:
                selected_datasets.remove(path)
            else:
                selected_datasets.append(path)
        except (ValueError, IndexError):
            print("[Error] Invalid selection.")
    return selected_datasets

def add_dataset_directory():
    print("\n--- Add New Dataset Directory ---")
    print("Registers a new dataset directory path for use in other toolkit functions.")
    print_cancel_message()
    
    path_str = get_input("Enter the absolute path to the dataset directory")
    if path_str == 'c' or not path_str: return

    path = Path(path_str)
    if path.is_dir():
        abs_path = str(path.resolve())
        if abs_path not in registered_datasets:
            registered_datasets.append(abs_path)
            print(f"\n[Success] Dataset '{abs_path}' added.")
        else:
            print(f"\n[Info] Dataset '{abs_path}' is already registered.")
    else:
        print(f"\n[Error] The path '{path_str}' is not a valid directory.")

def run_extract_from_bag():
    print("\n--- Extract Images from ROS Bag ---")
    print("Extracts images from a ROS2 bag file. Can run in fully automatic or interactive modes.")
    print_cancel_message()

    rosbag_dir = get_input("Enter path to ROS Bag directory")
    if rosbag_dir == 'c' or not rosbag_dir: return
    if not os.path.isdir(rosbag_dir): print(f"\n[Error] Directory not found: {rosbag_dir}"); return

    output_dir = get_input("Enter path for output dataset")
    if output_dir == 'c' or not output_dir: return
    
    mode = 0
    if GUI_ENABLED:
        mode_str = get_input("Select mode (0: All, 1: Interactive Single, 2: Interactive Range)", default="0")
        if mode_str == 'c': return
        try:
            mode = int(mode_str)
        except ValueError:
            print("[Error] Invalid mode."); return
    else:
        print("\n[Info] GUI not available. Extracting all images automatically (Mode 0).")

    workflow_params = config.get('workflow_parameters', {})
    topic = workflow_params.get('ros2_image_topic')
    fmts = workflow_params.get('image_format', 'png,jpg,jpeg').split(',')
    if not topic: print("\n[Error] 'ros2_image_topic' not defined in _config.yaml."); return

    data_handler.extract_images_from_rosbag(rosbag_dir, output_dir, topic, fmts, mode)

def run_labeling_tool():
    print("\n--- Launch Integrated Labeling Tool ---")
    print("A GUI tool for creating and reviewing bounding box labels.")
    print("Key-bindings: 1-9 to change class, W/E for Draw/Delete mode, A/D for prev/next image.")
    print_cancel_message()

    dataset_dir = get_dataset_from_user("Select a dataset to label/review")
    if dataset_dir == 'c' or not dataset_dir: return
    
    print("\nLaunching labeler... Close the labeling window to return to the menu.")
    labeling.launch_labeler(dataset_dir, config)

def run_split_dataset():
    print("\n--- Split Dataset for Training ---")
    print("Splits a labeled dataset into 'train' and 'val' subsets and creates a data.yaml file.")
    print_cancel_message()

    dataset_dir = get_dataset_from_user("Select a dataset to split")
    if dataset_dir == 'c' or not dataset_dir: return
    
    workflow_params = config.get('workflow_parameters', {})
    model_configs = config.get('model_configurations', {})
    ratio = workflow_params.get('train_split_ratio')
    classes = model_configs.get('classes')
    fmts = workflow_params.get('image_format', 'png,jpg,jpeg').split(',')

    if not all([ratio, classes]): print("\n[Error] Missing 'train_split_ratio' or 'classes' in _config.yaml."); return
    data_handler.split_dataset_for_training(dataset_dir, ratio, classes, fmts)

def run_unified_training():
    print("\n--- Train a Model ---")
    print("Trains a YOLO model using a selected dataset.")
    print("Key-bindings: Press 'q' during training to stop gracefully after the current epoch.")
    print_cancel_message()

    role_choice = get_input("What type of model to train? [1] Teacher, [2] Student")
    if role_choice == 'c': return
    if role_choice not in ['1', '2']: print("[Error] Invalid choice."); return
    
    role = 'teacher' if role_choice == '1' else 'student'
    model_config_name = f'{role}_model_config'
    
    dataset_path = get_dataset_from_user(f"Select a dataset for training the {role.capitalize()} model")
    if dataset_path == 'c' or not dataset_path: return
    
    model_configs = config.get('model_configurations', {})
    model_name = model_configs.get(model_config_name, {}).get('model_name')
    if not model_name: print(f"[Error] Could not find model name for '{model_config_name}' in config."); return

    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    run_name = f"{role}_{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nGenerated Run Name: {run_name}")
    
    exist_ok_str = get_input("Overwrite previous run with same name? (y/N)", default='n')
    if exist_ok_str == 'c': return
    exist_ok = exist_ok_str.lower() == 'y'
    
    training.train_yolo_model(dataset_path, model_config_name, role, run_name, config, exist_ok)

def run_auto_labeler():
    print("\n--- Auto-label a Dataset ---")
    print("Uses a trained Teacher model to automatically generate labels for an unlabeled dataset.")
    print_cancel_message()

    # Broader search for all .pt files in the teacher directory
    search_path = os.path.join('runs', 'train', 'teacher', '**', '*.pt')
    teacher_models = glob.glob(search_path, recursive=True)

    if not teacher_models:
        print("\n[Error] No trained Teacher models (.pt files) found in 'runs/train/teacher/'.")
        print("Please train a teacher model first or ensure weights files are in the correct directory.")
        return
    
    # Show a list and get user selection
    print("\nPlease select a Teacher model to use for labeling:")
    for i, model_path in enumerate(teacher_models, 1):
        print(f"  [{i}] {model_path}")
    
    weights_path = None
    while True:
        choice_str = get_input("Select a model (by number)")
        if choice_str == 'c': return
        try:
            choice_int = int(choice_str)
            if 1 <= choice_int <= len(teacher_models):
                weights_path = teacher_models[choice_int - 1]
                break
            else:
                print(f"[Error] Invalid selection.")
        except ValueError:
            print("[Error] Invalid input. Please enter a number.")
    
    dataset_path = get_dataset_from_user("Select a dataset to auto-label")
    if dataset_path == 'c' or not dataset_path: return
    
    print(f"\nUsing model: {weights_path}")
    labeling.auto_label_dataset(dataset_path, weights_path, config)

def run_merge_datasets():
    print("\n--- Merge Datasets ---")
    print("Combines multiple datasets into a single new dataset.")
    print_cancel_message()

    input_dirs = get_multiple_datasets_from_user()
    if 'c' in input_dirs: return
    if not input_dirs or len(input_dirs) < 2: print("\n[Error] Select at least two datasets."); return
    
    output_dir = get_input("Enter path for the new merged dataset")
    if output_dir == 'c' or not output_dir: return
    
    exist_ok_str = get_input(f"If '{output_dir}' exists, overwrite? (y/N)", default='n')
    if exist_ok_str == 'c': return
    exist_ok = exist_ok_str.lower() == 'y'

    fmts = config.get('workflow_parameters', {}).get('image_format', 'png,jpg,jpeg').split(',')
    data_handler.merge_datasets(input_dirs, output_dir, fmts, exist_ok)

def run_random_sampler():
    print("\n--- Random Sample from Dataset ---")
    print("Creates a new, smaller dataset by randomly sampling from a source dataset.")
    print_cancel_message()

    source_dir = get_dataset_from_user("Select a source dataset")
    if source_dir == 'c' or not source_dir: return
    
    output_dir = get_input("Enter path for the new sampled dataset")
    if output_dir == 'c' or not output_dir: return
    
    ratio_str = get_input("Enter sampling ratio (e.g., 0.1)")
    if ratio_str == 'c': return
    try:
        ratio = float(ratio_str)
    except ValueError: print("[Error] Invalid ratio."); return
    
    exist_ok_str = get_input(f"If '{output_dir}' exists, overwrite? (y/N)", default='n')
    if exist_ok_str == 'c': return
    exist_ok = exist_ok_str.lower() == 'y'

    fmts = config.get('workflow_parameters', {}).get('image_format', 'png,jpg,jpeg').split(',')
    data_handler.sample_dataset(source_dir, output_dir, ratio, fmts, exist_ok)

def main():
    global config
    check_environment()

    try:
        with open('_config.yaml', 'r') as f: config = yaml.safe_load(f)
    except FileNotFoundError:
        print("[WARNING] _config.yaml not found. Using default model settings.")
        config = {}
    except yaml.YAMLError as e: print(f"[Error] Failed to parse _config.yaml: {e}"); sys.exit(1)

    actions = {
        '1': run_extract_from_bag, '2': run_labeling_tool, '3': run_split_dataset,
        '4': run_unified_training, '5': run_auto_labeler, '6': run_merge_datasets,
        '7': run_random_sampler, '8': add_dataset_directory
    }
    
    while True:
        display_main_ui()
        choice = input("Select an option: ").strip()
        
        if choice == '0': print("Exiting toolkit. Goodbye!"); break
        
        if choice == '1' and not ROS2_ENABLED:
            print("\n[Error] This feature is disabled. Please install ROS2 and source the environment.")
            input("\nPress Enter to return...")
            continue
        if choice == '2' and not GUI_ENABLED:
            print("\n[Error] This feature is disabled in a non-GUI environment.")
            input("\nPress Enter to return...")
            continue

        action = actions.get(choice)
        if action:
            try:
                action()
                if choice != '8':
                    input("\nPress Enter to return to the main menu...")
            except Exception as e:
                print(f"\n[UNHANDLED ERROR] An unexpected error occurred: {e}")
                input("\nPress Enter to return...")
        else: 
            print(f"\n[Error] Invalid option '{choice}'.")
            input("\nPress Enter...")

if __name__ == "__main__":
    main()
