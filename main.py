import os
import sys
import glob
import yaml
from pathlib import Path
from datetime import datetime

# Import toolkit functions
from toolkit import data_handler, labeling, training
from toolkit.utils import build_class_hotkeys, normalize_class_map

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
        print("To enable it, install the extras via 'pip install -r requirements-for-ros2bag.txt',")
        print("ensure you have a valid ROS2 installation (e.g., Humble), and source its setup script.")
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

    raw_user_input = input(full_prompt)
    user_input = raw_user_input.strip()

    if user_input.lower() == 'c':
        return 'c'

    return user_input or default if default is not None else user_input

def get_sanitized_path_input(prompt, default=None, show_drag_tip=False):
    """Gets a path from user input, automatically stripping quotes."""
    if show_drag_tip:
        print("-> Tip: Drag a folder or file from your file explorer into this window to auto-fill the absolute path.")
    path_str = get_input(prompt, default)
    if path_str in ['c', '', None]:
        return path_str
    return path_str.replace('\'', '').replace('\"', '')


# --- Global State ---
registered_datasets = []
config = {}
BASE_DATASET_PATH = "datasets" # Base directory for new datasets
PROJECT_ROOT = Path(__file__).resolve().parent

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_header():
    print("=" * 80)
    print(" YOLO Dataset Studio ".center(80, "="))
    print("=" * 80)
    print()

def display_main_ui():
    clear_screen()
    display_header()

    # --- Registered Datasets Panel (Top) ---
    print(" Registered Datasets ".center(80, "-"))
    if not registered_datasets:
        print(" No datasets registered yet. Use option [9] to add one.")
    else:
        for i, path in enumerate(registered_datasets, 1):
            print(f"  <{i}> {path}")

    print()

    # --- Operation Option Title ---
    print(" Operation Options ".center(80, "-"))
    print()

    # --- Main Menu Panel (Bottom) ---
    menu = {
        "--- Data Preparation ---": {
            "1": "Extract Images from ROS Bag",
            "2": "Extract Frames from Video",
            "3": "Launch Integrated Labeling Tool",
            "4": "Split Dataset for Training",
        },
        "--- Training & Inference ---": {
            "5": "Train a Model (Teacher/Student)",
            "6": "Auto-label a Dataset with a Teacher",
        },
        "--- Utilities ---": {
            "7": "Merge Datasets",
            "8": "Sample from Dataset",
            "9": "Add New Dataset Directory",
        },
        "--- Exit ---": {
            "0": "Exit Studio",
        }
    }

    if not ROS2_ENABLED:
        menu["--- Data Preparation ---"]["1"] += " (Disabled: ROS2 Not Found)"
    if not GUI_ENABLED:
        if ROS2_ENABLED: # Avoid double-messaging if both are off
            menu["--- Data Preparation ---"]["1"] += " (Interactive Modes Disabled)"
        menu["--- Data Preparation ---"]["2"] += " (Interactive Modes Disabled)"
        menu["--- Data Preparation ---"]["3"] += " (Disabled: GUI Not Available)"

    for phase, options in menu.items():
        print(phase)
        for key, value in options.items():
            print(f"  [{key}] {value}")
        print() # Add a space after each section

    print("-" * 80)

def _prompt_for_new_dataset_path():
    """Helper function to prompt for, validate, and register a new dataset path."""
    print_cancel_message()
    while True:
        path_str = get_sanitized_path_input("Enter the absolute path to the dataset directory", show_drag_tip=True)

        if path_str == 'c':
            return 'c'
        if not path_str:
            print("[Error] Path cannot be empty.")
            continue

        path = Path(path_str)
        if path.is_dir():
            abs_path = str(path.resolve())
            if abs_path not in registered_datasets:
                registered_datasets.append(abs_path)
                print(f"\n[Success] Dataset '{abs_path}' has been registered and selected.")
            else:
                print(f"\n[Info] Dataset '{abs_path}' was already registered. Selecting it.")
            return abs_path
        else:
            print(f"\n[Error] The path '{path_str}' is not a valid directory. Please try again.")

def get_dataset_from_user(prompt="Select a dataset to use (by number)"):
    # Case 1: No datasets are registered yet.
    if not registered_datasets:
        print("\n[Info] No datasets are registered yet. Please provide a path directly.")
        return _prompt_for_new_dataset_path()

    # Case 2: Datasets exist, prompt for selection.
    while True:
        # Update prompt to include 'add' instruction
        full_prompt = f"{prompt} or enter 'add' for a new path"
        choice_str = get_input(full_prompt)
        
        if choice_str == 'c': return 'c'

        if choice_str.lower() == 'add':
            print("\n--- Add New Dataset Path ---")
            new_path = _prompt_for_new_dataset_path()
            # If a new path was successfully added, return it for immediate use.
            # If the user cancelled ('c'), the loop will continue after returning 'c'.
            if new_path != 'c':
                return new_path
            # If user cancels adding a new path, we just re-prompt for selection.
            # We can't return 'c' here as it would cancel the parent operation.
            # Instead, we let the loop start over.
            print("\nReturning to dataset selecti==on...")
            # Re-display the (potentially updated) list
            print("\n--- Registered Datasets ---")
            for i, path in enumerate(registered_datasets, 1):
                print(f"  <{i}> {path}")
            print("-" * 27)
            continue # Re-start the selection loop

        try:
            choice_int = int(choice_str)
            
            # Option A: User selects an existing dataset from the list.
            if 1 <= choice_int <= len(registered_datasets):
                return registered_datasets[choice_int - 1]
            else:
                print(f"[Error] Invalid selection. Please enter a number between 1 and {len(registered_datasets)}.")
        except ValueError:
            print("[Error] Invalid input. Please enter a number from the list or 'add'.")

def get_multiple_datasets_from_user():
    if not registered_datasets:
        print("\n[Info] No datasets registered yet. Add a dataset path to begin.")

    selected_datasets: list[str] = []

    while True:
        print("\n--- Select datasets to merge ---")
        if not registered_datasets:
            print("  (no registered datasets)")
        else:
            for i, path in enumerate(registered_datasets, 1):
                marker = "*" if path in selected_datasets else " "
                print(f"  <{i}>[{marker}] {path}")

        print("\nInstructions: Enter dataset numbers to toggle (e.g., '1 3'), 'add' to register a new path, 'done' or Enter to finish, or 'c' to cancel.")
        choice_str = get_input("Selection")

        if choice_str == 'c':
            return 'c'

        if choice_str.strip() == '' or choice_str.lower() == 'done':
            break

        if choice_str.lower() == 'add':
            new_path = _prompt_for_new_dataset_path()
            if new_path not in ('c', None):
                if new_path not in selected_datasets:
                    selected_datasets.append(new_path)
            continue

        tokens = [token for token in choice_str.replace(',', ' ').split() if token]
        if not tokens:
            print("[Error] Please enter a valid selection.")
            continue

        valid = True
        for token in tokens:
            try:
                index = int(token) - 1
                if index < 0:
                    raise ValueError
                path = registered_datasets[index]
            except (ValueError, IndexError):
                print(f"[Error] Invalid selection '{token}'.")
                valid = False
                break

            if path in selected_datasets:
                selected_datasets.remove(path)
            else:
                selected_datasets.append(path)

        if not valid:
            continue

    return selected_datasets

def add_dataset_directory():
    print("\n--- Add New Dataset Directory ---")
    print("Registers a new dataset directory path for use in other toolkit functions.")
    
    # Use the new helper function for consistency. 
    # The return value isn't needed here, but it reuses the same validated input logic.
    new_path = _prompt_for_new_dataset_path()
    if new_path != 'c':
        # The helper function already prints success, so we just wait for user input to return.
        pass


def run_extract_from_bag():
    print("\n--- Extract Images from ROS Bag ---")
    print("Extracts images from a ROS2 bag file. Can run in fully automatic or interactive modes.")
    print_cancel_message()

    rosbag_dir = get_sanitized_path_input("Enter path to ROS Bag directory", show_drag_tip=True)
    if rosbag_dir == 'c' or not rosbag_dir: return
    if not os.path.isdir(rosbag_dir): print(f"\n[Error] Directory not found: {rosbag_dir}"); return

    dataset_name = get_input("Enter a name for the new output dataset")
    if dataset_name == 'c' or not dataset_name: return

    os.makedirs(BASE_DATASET_PATH, exist_ok=True)
    output_dir = os.path.join(BASE_DATASET_PATH, dataset_name)
    print(f"-> Output will be saved to: {output_dir}")

    topic = get_input("Enter the ROS2 image topic to extract", default="/camera/image_raw")
    if topic == 'c' or not topic: return

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
    fmts = workflow_params.get('image_format', 'png,jpg,jpeg').split(',')

    data_handler.extract_images_from_rosbag(rosbag_dir, output_dir, topic, fmts, mode)


def run_extract_from_video():
    print("\n--- Extract Frames from Video ---")
    print("Convert standard video files (e.g., mp4, avi, mov, mkv) into YOLO-ready image datasets.")
    print_cancel_message()

    video_path = get_sanitized_path_input("Enter path to the video file", show_drag_tip=True)
    if video_path == 'c' or not video_path:
        return
    if not os.path.isfile(video_path):
        print(f"\n[Error] File not found: {video_path}")
        return

    dataset_name = get_input("Enter a name for the new output dataset")
    if dataset_name == 'c' or not dataset_name:
        return

    os.makedirs(BASE_DATASET_PATH, exist_ok=True)
    output_dir = os.path.join(BASE_DATASET_PATH, dataset_name)
    print(f"-> Output will be saved to: {output_dir}")

    mode = 0
    if GUI_ENABLED:
        mode_str = get_input("Select mode (0: All, 1: Interactive Single, 2: Interactive Range)", default="0")
        if mode_str == 'c':
            return
        try:
            mode = int(mode_str)
            if mode not in [0, 1, 2]:
                raise ValueError
        except ValueError:
            print("[Error] Invalid mode.")
            return
    else:
        print("\n[Info] GUI not available. Extracting all frames automatically (Mode 0).")

    workflow_params = config.get('workflow_parameters', {})
    fmts = workflow_params.get('image_format', 'png,jpg,jpeg').split(',')

    data_handler.extract_frames_from_video(video_path, output_dir, fmts, mode)

def run_labeling_tool():
    print("\n--- Launch Integrated Labeling Tool ---")
    print("A GUI tool for creating and reviewing bounding box labels.")

    # Display class information and key-bindings
    classes = config.get('model_configurations', {}).get('classes', {})
    class_hotkeys = build_class_hotkeys(classes)

    if not classes:
        print("\n[Warning] No classes defined in settings.yaml.")
    else:
        print("\n--- Class & Key-bindings ---")
        colors = {c: ((c*55+50)%256, (c*95+100)%256, (c*135+150)%256) for c in classes.keys()}
        print(f"{'Key':<5} {'ID':<5} {'Name':<25} {'Color (BGR)'}")
        print("-"*65)
        for key, class_id, name in class_hotkeys:
            print(f"{key:<5} {class_id:<5} {name:<25} {colors.get(class_id)}")
        if class_hotkeys and len(class_hotkeys) < len(classes):
            print("(Only the first 9 classes are mapped to number keys 1-9.)")
        print("-"*65)

    print("Other Keys: (W) Draw | (E) Delete | (A) Prev | (D) Next | (I) Toggle Class Names | (F) Flag for Review | (Q) Quit")
    print_cancel_message()

    dataset_dir = get_dataset_from_user("Select a dataset to label/review")
    if dataset_dir == 'c' or not dataset_dir: return

    print("\nLaunching labeler... Close the labeling window to return to the menu.")
    labeling.launch_labeler(dataset_dir, config)

def get_split_ratios_from_user(expected_subsets=None):
    """Interactively prompts the user to enter split ratios that sum to 10."""
    subset_sequence = list(expected_subsets) if expected_subsets else None

    while True:
        if subset_sequence:
            names = ", ".join(subset_sequence)
            prompt = f"Enter split ratios for {names} (sum must be 10). Example: '7 3'"
        else:
            prompt = "Enter split ratios for train, val, and optionally test (e.g., '7 3' or '6 2 2'). The sum must be 10."

        ratios_str = get_input(prompt)
        if ratios_str == 'c':
            return 'c'

        try:
            parts = [int(p) for p in ratios_str.split()]
        except ValueError:
            print("\n[Error] Invalid input. Please enter space-separated integers.")
            continue

        if subset_sequence:
            if len(parts) != len(subset_sequence):
                print(f"\n[Error] Expected {len(subset_sequence)} values for {', '.join(subset_sequence)}.")
                continue
            if sum(parts) != 10:
                print(f"\n[Error] The sum of the ratios must be 10, but got {sum(parts)}. Please try again.")
                continue
            return {subset_sequence[i]: parts[i] for i in range(len(subset_sequence))}

        if sum(parts) != 10:
            print(f"\n[Error] The sum of the ratios must be 10, but got {sum(parts)}. Please try again.")
            continue

        if len(parts) == 2:
            return {'train': parts[0], 'val': parts[1]}
        if len(parts) == 3:
            return {'train': parts[0], 'val': parts[1], 'test': parts[2]}

        print("\n[Error] Please enter 2 or 3 numbers (e.g., for train/val or train/val/test). Please try again.")

def run_split_dataset():
    print("\n--- Split Dataset for Training ---")
    print("Splits a labeled dataset into 'train', 'val' (and optionally 'test') subsets and creates a data.yaml file.")
    print_cancel_message()

    dataset_dir = get_dataset_from_user("Select a dataset to split")
    if dataset_dir == 'c' or not dataset_dir:
        return

    destination_choice = None
    while destination_choice is None:
        dest_input = get_input(
            "Save split dataset to which location? [1] Overwrite selected dataset, [2] Create new dataset under 'datasets/'",
            default='1'
        )
        if dest_input == 'c':
            return
        if dest_input in ('1', '2'):
            destination_choice = dest_input
        else:
            print("[Error] Invalid selection. Please enter '1' or '2'.")

    output_dir = None
    exist_ok = False
    if destination_choice == '2':
        os.makedirs(BASE_DATASET_PATH, exist_ok=True)
        source_path = Path(dataset_dir).resolve()

        while True:
            dataset_name = get_input("Enter a name for the new split dataset")
            if dataset_name == 'c':
                return
            if not dataset_name:
                print("[Error] Dataset name cannot be empty.")
                continue

            candidate_dir = Path(BASE_DATASET_PATH) / dataset_name
            candidate_resolved = candidate_dir.resolve()
            if candidate_resolved == source_path:
                print("[Error] New dataset path matches the source dataset. Choose a different name or select option [1].")
                continue

            if candidate_dir.exists():
                overwrite_choice = get_input(
                    f"Directory '{candidate_dir}' exists. Overwrite? (y/N)",
                    default='n'
                )
                if overwrite_choice == 'c':
                    return
                if overwrite_choice.lower() == 'y':
                    exist_ok = True
                    output_dir = str(candidate_dir)
                    break
                print("Please choose a different name.")
                continue

            output_dir = str(candidate_dir)
            break

        if output_dir:
            print(f"-> Split dataset will be saved to: {output_dir}")

    ratios = get_split_ratios_from_user()
    if ratios == 'c':
        print("\nOperation cancelled.")
        return

    model_configs = config.get('model_configurations', {})
    workflow_params = config.get('workflow_parameters', {})
    classes = model_configs.get('classes')
    fmts = workflow_params.get('image_format', 'png,jpg,jpeg').split(',')

    if not classes:
        print("\n[Error] Missing 'classes' in settings.yaml.")
        return

    data_handler.split_dataset_for_training(
        dataset_dir,
        ratios,
        classes,
        fmts,
        output_dir=output_dir,
        exist_ok=exist_ok
    )

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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{role}_{model_name}_{timestamp}_{dataset_name}"
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


def _print_merge_dataset_summary(profiles):
    print("\n--- Source Dataset Summary ---")
    for idx, profile in enumerate(profiles, 1):
        ratio_map = profile.ratio_map()
        subset_descriptions = []
        for subset in data_handler.STANDARD_SUBSETS:
            if subset in profile.subset_pairs:
                count = len(profile.subset_pairs[subset])
                percent = ratio_map.get(subset, 0.0) * 100
                subset_descriptions.append(f"{subset}: {count} ({percent:.1f}%)")

        print(f"[{idx}] {profile.path}")
        print(f"    structure: {profile.structure}")
        print(f"    total pairs: {profile.total_pairs}")
        if subset_descriptions:
            print(f"    splits: {', '.join(subset_descriptions)}")
        if profile.missing_labels:
            print(f"    missing labels ignored: {profile.missing_labels}")
        if profile.unassigned_pairs:
            print(f"    pairs outside standard splits: {len(profile.unassigned_pairs)}")
        if profile.extra_subsets:
            extras = ', '.join(f"{name} ({count})" for name, count in profile.extra_subsets.items())
            print(f"    additional split names detected: {extras}")


def _compute_profile_ratio(profile, target_subsets):
    ratio_map = profile.ratio_map()
    return {subset: ratio_map.get(subset, 0.0) for subset in target_subsets}


def _ratios_close(a, b, tolerance=0.02):
    for key in a.keys():
        if abs(a.get(key, 0.0) - b.get(key, 0.0)) > tolerance:
            return False
    return True


def _normalize_ratio_dict(ratio_dict):
    total = sum(ratio_dict.values())
    if total <= 0:
        return {k: 0.0 for k in ratio_dict}
    return {k: value / total for k, value in ratio_dict.items()}


def _prompt_ratio_plan(profiles, target_subsets, include_test):
    while True:
        choice = get_input("Preserve existing split ratios from source datasets? (y/N)", default='y')
        if choice == 'c':
            return 'c'

        keep_existing = choice.lower() in ('y', 'yes') or choice == ''

        if keep_existing:
            if any(profile.unassigned_pairs for profile in profiles):
                print("\n[Error] Some datasets contain images that are not assigned to 'train', 'val', or 'test'. Configure new ratios instead.")
                continue

            if include_test and not all(profile.has_subset('test') for profile in profiles):
                print("\n[Error] Not all datasets include a 'test' split. Configure new ratios or exclude 'test' in the merged dataset.")
                continue

            if not include_test and any(profile.has_subset('test') for profile in profiles):
                print("\n[Error] Some datasets include a 'test' split. Configure new ratios to redistribute those samples.")
                continue

            reference = _compute_profile_ratio(profiles[0], target_subsets)
            ratios_match = all(
                _ratios_close(reference, _compute_profile_ratio(profile, target_subsets))
                for profile in profiles[1:]
            )

            if not ratios_match:
                print("\n[Error] Source datasets use differing split ratios. Configure new ratios instead.")
                continue

            return {'mode': 'preserve', 'reference': reference}

        uniform_choice = get_input("Apply the same split ratios to all datasets? (y/N)", default='y')
        if uniform_choice == 'c':
            return 'c'

        apply_uniform = uniform_choice.lower() in ('y', 'yes') or uniform_choice == ''

        if apply_uniform:
            ratio_values = get_split_ratios_from_user(target_subsets)
            if ratio_values == 'c':
                continue
            normalized = _normalize_ratio_dict(ratio_values)
            return {'mode': 'uniform', 'ratios': normalized}

        per_dataset = {}
        for profile in profiles:
            print(f"\n--- Configure ratios for {profile.path} ---")
            ratio_values = get_split_ratios_from_user(target_subsets)
            if ratio_values == 'c':
                per_dataset = None
                break
            per_dataset[profile.path] = _normalize_ratio_dict(ratio_values)

        if per_dataset is None:
            continue

        return {'mode': 'per_dataset', 'ratios': per_dataset}


def run_merge_datasets():
    print("\n--- Merge Datasets ---")
    print("Combines multiple datasets into a single new dataset.")
    print_cancel_message()

    input_dirs = get_multiple_datasets_from_user()
    if input_dirs == 'c':
        return
    if not input_dirs or len(input_dirs) < 2:
        print("\n[Error] Select at least two datasets to merge.")
        return

    fmts = config.get('workflow_parameters', {}).get('image_format', 'png,jpg,jpeg').split(',')
    profiles = [data_handler.survey_dataset_for_merge(path, fmts) for path in input_dirs]

    if any(profile.total_pairs == 0 for profile in profiles):
        print("\n[Error] One or more datasets do not contain labeled images under an 'images' directory.")
        return

    _print_merge_dataset_summary(profiles)

    include_test_default = 'y' if all(profile.has_subset('test') for profile in profiles) else 'n'
    while True:
        include_test_input = get_input("Include 'test' split in merged dataset? (y/N)", default=include_test_default)
        if include_test_input == 'c':
            return
        if include_test_input.lower() in ('y', 'yes', 'n', 'no', ''):
            include_test = include_test_input.lower() in ('y', 'yes') or include_test_input == '' and include_test_default == 'y'
            break
        print("[Error] Please respond with 'y' or 'n'.")

    target_subsets = ['train', 'val'] + (['test'] if include_test else [])

    subset_preview = ', '.join(target_subsets)
    print("\nSelect output directory structure:")
    print(f"  [1] images/{{{subset_preview}}} and labels/{{{subset_preview}}}")
    print(f"  [2] {{{subset_preview}}}/images and {{{subset_preview}}}/labels")

    while True:
        structure_choice = get_input("Structure option", default='1')
        if structure_choice == 'c':
            return
        if structure_choice in ('1', '2', ''):
            break
        print("[Error] Please enter '1' or '2'.")

    target_structure = 'images_first' if structure_choice in ('1', '') else 'subset_first'

    ratio_plan = _prompt_ratio_plan(profiles, target_subsets, include_test)
    if ratio_plan == 'c':
        return

    dataset_name = get_input("\nEnter a name for the new merged dataset")
    if dataset_name == 'c' or not dataset_name:
        return

    os.makedirs(BASE_DATASET_PATH, exist_ok=True)
    output_dir = os.path.join(BASE_DATASET_PATH, dataset_name)
    print(f"-> Merged dataset will be saved to: {output_dir}")

    exist_ok_str = get_input(f"If '{output_dir}' exists, overwrite? (y/N)", default='n')
    if exist_ok_str == 'c':
        return
    exist_ok = exist_ok_str.lower() == 'y'

    merge_config = {
        'profiles': profiles,
        'target_structure': target_structure,
        'target_subsets': target_subsets,
        'ratio_plan': ratio_plan,
        'include_test': include_test,
    }

    data_handler.merge_datasets(merge_config, output_dir, exist_ok=exist_ok)

def run_random_sampler():
    print("\n--- Sample from Dataset ---")
    print("Creates a new, smaller dataset by sampling from a source dataset (randomly or uniformly).")
    print_cancel_message()

    source_dir = get_dataset_from_user("Select a source dataset")
    if source_dir == 'c' or not source_dir: return

    dataset_name = get_input("Enter a name for the new sampled dataset")
    if dataset_name == 'c' or not dataset_name: return
    
    os.makedirs(BASE_DATASET_PATH, exist_ok=True)
    output_dir = os.path.join(BASE_DATASET_PATH, dataset_name)
    print(f"-> Sampled dataset will be saved to: {output_dir}")

    fmts = config.get('workflow_parameters', {}).get('image_format', 'png,jpg,jpeg').split(',')

    # Get total image count
    all_image_data = data_handler.get_all_image_data(source_dir, fmts)
    total_images = len(all_image_data)

    if total_images == 0:
        print(f"[Error] No images found in the selected dataset: {source_dir}")
        return

    print(f"\nTotal images in dataset: {total_images}")

    desired_samples_str = get_input(f"Enter desired number of samples (1-{total_images})")
    if desired_samples_str == 'c': return

    try:
        desired_samples = int(desired_samples_str)
        if not (1 <= desired_samples <= total_images):
            print(f"[Error] Desired samples must be between 1 and {total_images}.")
            return
    except ValueError:
        print("[Error] Invalid number of samples.")
        return

    ratio = desired_samples / total_images
    print(f"Calculated sampling ratio: {ratio:.4f}")

    method_choice = get_input("Select sampling method [1] Random, [2] Uniform (distributed)", default='1')
    if method_choice == 'c': return
    method = 'uniform' if method_choice == '2' else 'random'

    exist_ok_str = get_input(f"If '{output_dir}' exists, overwrite? (y/N)", default='n')
    if exist_ok_str == 'c': return
    exist_ok = exist_ok_str.lower() == 'y'

    data_handler.sample_dataset(source_dir, output_dir, ratio, fmts, exist_ok, method)

def main():
    global config
    check_environment()

    # Create the base datasets directory on startup if it doesn't exist
    os.makedirs(BASE_DATASET_PATH, exist_ok=True)

    search_candidates = [
        Path.cwd() / "settings.yaml",
        PROJECT_ROOT / "settings.yaml",
        Path.cwd() / "models_config.yaml",
        PROJECT_ROOT / "models_config.yaml",
    ]

    config_path = next((c for c in search_candidates if c.exists()), None)

    if not config_path:
        print("[WARNING] settings.yaml not found. Using default model settings.")
        config = {}
    else:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"[Error] Failed to parse {config_path}: {e}")
            sys.exit(1)

        if config_path.name == "models_config.yaml":
            print("[INFO] Loaded legacy models_config.yaml. Please rename it to settings.yaml.")

    if not isinstance(config, dict):
        config = {}

    model_configs = config.setdefault('model_configurations', {})
    model_configs['classes'] = normalize_class_map(model_configs.get('classes'))

    actions = {
        '1': run_extract_from_bag,
        '2': run_extract_from_video,
        '3': run_labeling_tool,
        '4': run_split_dataset,
        '5': run_unified_training,
        '6': run_auto_labeler,
        '7': run_merge_datasets,
        '8': run_random_sampler,
        '9': add_dataset_directory
    }

    while True:
        display_main_ui()
        choice = input("Select an option: ").strip()

        if choice == '0': print("Exiting studio. Goodbye!"); break

        if choice == '1' and not ROS2_ENABLED:
            print("\n[Error] This feature is disabled. Please install ROS2 and source the environment.")
            input("\nPress Enter to return...")
            continue
        if choice == '3' and not GUI_ENABLED:
            print("\n[Error] This feature is disabled in a non-GUI environment.")
            input("\nPress Enter to return...")
            continue

        action = actions.get(choice)
        if action:
            try:
                action()
                if choice != '9':
                    input("\nPress Enter to return to the main menu...")
            except Exception as e:
                print(f"\n[UNHANDLED ERROR] An unexpected error occurred: {e}")
                input("\nPress Enter to return...")
        else:
            print(f"\n[Error] Invalid option '{choice}'.")
            input("\nPress Enter...")

if __name__ == "__main__":
    main()
