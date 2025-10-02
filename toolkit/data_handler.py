import os
import glob
import shutil
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import subprocess
import threading
import time
import sys
import yaml
from tqdm import tqdm

# 이 파일이 toolkit 폴더 안에 있다고 가정하고, utils.py를 찾기 위함입니다.
# 만약 구조가 다르다면 이 부분을 수정해야 할 수 있습니다.
try:
    from toolkit.utils import get_label_path
except ImportError:
    # utils 모듈을 찾을 수 없을 경우를 대비한 임시 함수
    def get_label_path(image_path):
        label_path = str(image_path).replace('images', 'labels', 1)
        base, _ = os.path.splitext(label_path)
        return base + '.txt'


STANDARD_SUBSETS: Tuple[str, ...] = ('train', 'val', 'test')


@dataclass
class MergeDatasetProfile:
    path: str
    structure: str
    subset_pairs: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    unassigned_pairs: List[Tuple[str, str]] = field(default_factory=list)
    total_pairs: int = 0
    missing_labels: int = 0
    extra_subsets: Dict[str, int] = field(default_factory=dict)

    def available_subsets(self) -> List[str]:
        return [name for name, pairs in self.subset_pairs.items() if pairs]

    def has_subset(self, name: str) -> bool:
        return bool(self.subset_pairs.get(name))

    def ratio_map(self) -> Dict[str, float]:
        if not self.total_pairs:
            return {name: 0.0 for name in self.subset_pairs.keys()}
        return {name: len(pairs) / self.total_pairs for name, pairs in self.subset_pairs.items()}


def _detect_dataset_structure(dataset_path: Path) -> str:
    images_dir = dataset_path / 'images'
    has_images_root = images_dir.is_dir()
    images_subdirs = {child.name for child in images_dir.iterdir()} if has_images_root else set()
    has_subset_first = any((dataset_path / subset / 'images').is_dir() for subset in STANDARD_SUBSETS)
    has_images_first = any(sub in STANDARD_SUBSETS for sub in images_subdirs)

    if has_images_first and has_subset_first:
        return 'mixed'
    if has_images_first:
        return 'images_first'
    if has_subset_first:
        return 'subset_first'
    if has_images_root:
        return 'images_flat'
    return 'unknown'


def _detect_subset_name(image_path: Path) -> Optional[str]:
    parts = image_path.parts
    for idx, part in enumerate(parts):
        if part != 'images':
            continue
        if idx + 1 < len(parts) and parts[idx + 1] in STANDARD_SUBSETS:
            return parts[idx + 1]
        if idx - 1 >= 0 and parts[idx - 1] in STANDARD_SUBSETS:
            return parts[idx - 1]
    return None


def survey_dataset_for_merge(dataset_dir: str, image_formats: List[str]) -> MergeDatasetProfile:
    dataset_path = Path(dataset_dir)
    subset_pairs: Dict[str, List[Tuple[str, str]]] = {name: [] for name in STANDARD_SUBSETS}
    unassigned_pairs: List[Tuple[str, str]] = []
    extra_subsets: Dict[str, int] = {}
    missing_labels = 0

    normalized_formats = [fmt.strip().lstrip('.') for fmt in image_formats if fmt.strip()]
    image_paths = []
    for fmt in normalized_formats:
        image_paths.extend(dataset_path.glob(f'**/*.{fmt}'))

    seen_images = set()
    for img_path in sorted(image_paths):
        if 'images' not in img_path.parts:
            continue
        str_img = str(img_path)
        if str_img in seen_images:
            continue
        seen_images.add(str_img)

        label_path = get_label_path(str_img)
        if not os.path.exists(label_path):
            missing_labels += 1
            continue

        subset = _detect_subset_name(img_path)
        if subset in subset_pairs:
            subset_pairs[subset].append((str_img, label_path))
        elif subset is None:
            unassigned_pairs.append((str_img, label_path))
        else:
            extra_subsets[subset] = extra_subsets.get(subset, 0) + 1
            unassigned_pairs.append((str_img, label_path))

    total_pairs = sum(len(pairs) for pairs in subset_pairs.values()) + len(unassigned_pairs)
    structure = _detect_dataset_structure(dataset_path)

    profile = MergeDatasetProfile(
        path=str(dataset_path),
        structure=structure,
        subset_pairs=subset_pairs,
        unassigned_pairs=unassigned_pairs,
        total_pairs=total_pairs,
        missing_labels=missing_labels,
        extra_subsets=extra_subsets,
    )
    return profile

def _get_topic_type(bag_dir, topic_name):
    metadata_path = os.path.join(bag_dir, 'metadata.yaml')
    if not os.path.exists(metadata_path): return None
    with open(metadata_path, 'r') as f: metadata = yaml.safe_load(f)['rosbag2_bagfile_information']
    for topic_info in metadata['topics_with_message_count']:
        if topic_info['topic_metadata']['name'] == topic_name: return topic_info['topic_metadata']['type']
    return None

class RosBagPlayer:
    """
    Manages ROS2 bag playback using the native 'ros2 bag play' command
    and controls it via ROS2 services for robust pause/resume functionality.
    """
    def __init__(self, image_topic):
        import rclpy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        from rosbag2_interfaces.srv import TogglePaused

        self.rclpy = rclpy
        if not self.rclpy.ok():
            self.rclpy.init()

        self.node = self.rclpy.create_node('yolo_toolkit_player_controller_node')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.bag_process = None
        self.is_paused = True  # Start in paused state

        self.TogglePausedSrv = TogglePaused
        self.subscription = self.node.create_subscription(
            Image, image_topic, self._image_callback, 10)
        self.toggle_client = self.node.create_client(self.TogglePausedSrv, '/rosbag2_player/toggle_paused')
        
        self.ros_thread = threading.Thread(target=self.rclpy.spin, args=(self.node,), daemon=True)
        self.ros_thread.start()
        print("ROS2 subscriber and service client node started.")

    def _image_callback(self, msg):
        with self.frame_lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def play_bag(self, bag_path):
        # Use the built-in '--start-paused' option for reliable startup
        command = ['ros2', 'bag', 'play', bag_path, '--start-paused']
        print(f"Executing command: {' '.join(command)}")
        try:
            self.bag_process = subprocess.Popen(command)
            if not self.toggle_client.wait_for_service(timeout_sec=5.0):
                print("[Error] Could not connect to /rosbag2_player/toggle_paused service.")
                self.cleanup()
                return False
            
            print("Player started in paused state.")
            return True
        except FileNotFoundError:
            print("[Error] 'ros2' command not found. Is ROS2 sourced correctly?")
            return False
        except Exception as e:
            print(f"[Error] Failed to start 'ros2 bag play': {e}")
            return False

    def toggle_pause(self):
        if not self.toggle_client.service_is_ready():
            print("[Warning] Toggle service not available.")
            return
        
        self.is_paused = not self.is_paused
        self.toggle_client.call_async(self.TogglePausedSrv.Request())

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def cleanup(self):
        print("Cleaning up resources...")
        if self.bag_process and self.bag_process.poll() is None:
            self.bag_process.terminate()
            self.bag_process.wait()
            print("ROS2 bag play process terminated.")
        if self.rclpy.ok():
            self.node.destroy_node()

def extract_images_from_rosbag(rosbag_dir, output_dir, image_topic, image_formats, mode=0):
    """
    Extracts images from a ROS2 bag file.
    Mode 0: Non-interactive, uses SequentialReader for speed.
    Modes 1 & 2: Interactive GUI, uses native 'ros2 bag play' for smooth playback.
    """
    try:
        import cv2
        if mode == 0:
            from cv_bridge import CvBridge
            from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
            from rclpy.serialization import deserialize_message
            from rosidl_runtime_py.utilities import get_message
        else:
             import rclpy
    except ImportError:
        print("[Error] ROS2/OpenCV libraries not found. Cannot proceed with extraction.")
        return False

    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    start_index = 0
    existing = [int(os.path.splitext(f)[0]) for f in os.listdir(images_dir) if f.split('.')[-1] in image_formats and f.split('.')[0].isdigit()]
    if existing:
        start_index = max(existing) + 1
    
    saved_count = 0

    if mode == 0:
        print("Running non-interactive extraction...")
        reader = SequentialReader()
        try:
            reader.open(StorageOptions(uri=rosbag_dir, storage_id='sqlite3'), ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
        except Exception as e:
            print(f"[Error] Failed to open ROS Bag: {e}")
            return False
        topic_type_str = _get_topic_type(rosbag_dir, image_topic)
        if not topic_type_str:
            print(f"[Error] Topic '{image_topic}' not found in the bag file.")
            return False
        msg_type = get_message(topic_type_str)
        bridge = CvBridge()
        total_messages = 0
        metadata_path = os.path.join(rosbag_dir, 'metadata.yaml')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)['rosbag2_bagfile_information']
            for topic_info in metadata['topics_with_message_count']:
                if topic_info['topic_metadata']['name'] == image_topic:
                    total_messages = topic_info['message_count']
        
        pbar = tqdm(total=total_messages, desc="Extracting from Bag")
        for topic, data, t in reader:
            if topic == image_topic:
                try:
                    cv_image = bridge.imgmsg_to_cv2(deserialize_message(data, msg_type), "bgr8")
                    fname = f"{start_index:06d}.{image_formats[0]}"
                    cv2.imwrite(os.path.join(images_dir, fname), cv_image)
                    saved_count += 1
                    start_index += 1
                except Exception as e:
                    print(f"\n[Warning] Could not process a message: {e}")
                finally:
                    if total_messages > 0: pbar.update(1)
        if total_messages > 0: pbar.close()
        print(f"\nExtraction finished. {saved_count} images saved.")
        return True

    # --- Interactive extraction (modes 1 and 2) ---
    player = RosBagPlayer(image_topic)
    
    is_saving, save_single = False, False

    def mouse_callback(event, x, y, flags, param):
        nonlocal save_single, is_saving
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == 1: save_single = True
            elif mode == 2: is_saving = not is_saving
    
    cv2.namedWindow("ROS2 Bag Player")
    cv2.setMouseCallback("ROS2 Bag Player", mouse_callback)
    print("\n--- Interactive Player Controls ---")
    print("  Spacebar: Play/Pause")
    print("  Mouse Click: Save image (single or toggle range based on mode)")
    print("  Q: Quit")
    print("---------------------------------")

    if not player.play_bag(rosbag_dir):
        player.cleanup()
        cv2.destroyAllWindows()
        return False

    while True:
        if player.bag_process and player.bag_process.poll() is not None:
            print("\nROS2 bag play process has ended.")
            break
            
        frame = player.get_frame()

        display_image = None
        if frame is None:
            # If no frame has been received yet, show a placeholder
            display_image = np.zeros((480, 640, 3), dtype=np.uint8)
            text = "Press SPACE to play"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (display_image.shape[1] - text_size[0]) // 2
            text_y = (display_image.shape[0] + text_size[1]) // 2
            cv2.putText(display_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # A frame has been received, use it
            if (mode == 1 and save_single) or (mode == 2 and is_saving):
                fname = f"{start_index:06d}.{image_formats[0]}"
                cv2.imwrite(os.path.join(images_dir, fname), frame)
                saved_count += 1
                start_index += 1
                save_single = False

            display_image = frame.copy()
            if is_saving and mode == 2:
                cv2.circle(display_image, (30, 30), 20, (0, 0, 255), -1)
                cv2.putText(display_image, "REC", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if player.is_paused:
                cv2.putText(display_image, "PAUSED", (display_image.shape[1] - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("ROS2 Bag Player", display_image)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            player.toggle_pause()

    cv2.destroyAllWindows()
    player.cleanup()
    print(f"\nExtraction finished. {saved_count} images saved.")
    return True

def extract_frames_from_video(video_path, output_dir, image_formats, mode=0):
    """Extracts frames from a video file with optional interactive selection modes."""
    try:
        import cv2
    except ImportError:
        print("[Error] OpenCV libraries not found. Cannot process video files.")
        return False

    if not os.path.isfile(video_path):
        print(f"[Error] Video file not found: {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video file: {video_path}")
        return False

    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    normalized_formats = [fmt.strip().lower() for fmt in image_formats if fmt and fmt.strip()]
    default_ext = normalized_formats[0] if normalized_formats else 'png'
    if default_ext not in normalized_formats and default_ext:
        normalized_formats.append(default_ext)

    existing = [
        int(os.path.splitext(f)[0])
        for f in os.listdir(images_dir)
        if f.split('.')[-1].lower() in normalized_formats and f.split('.')[0].isdigit()
    ]
    start_index = max(existing) + 1 if existing else 0
    saved_count = 0

    def save_frame(frame):
        nonlocal start_index, saved_count
        filename = f"{start_index:06d}.{default_ext}"
        frame_path = os.path.join(images_dir, filename)
        if not cv2.imwrite(frame_path, frame):
            print(f"[Warning] Failed to write frame to '{frame_path}'.")
            return
        saved_count += 1
        start_index += 1

    if mode == 0:
        print("Running non-interactive extraction...")
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        pbar = tqdm(total=frame_total, desc="Extracting Frames") if frame_total else None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            save_frame(frame)
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
        cap.release()
        print(f"\nExtraction finished. {saved_count} images saved.")
        return True

    window_name = "Video Frame Extractor"
    cv2.namedWindow(window_name)

    is_saving = False
    save_single = False
    paused = True
    current_frame = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 30

    def mouse_callback(event, _x, _y, _flags, _param):
        nonlocal save_single, is_saving
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == 1:
                save_single = True
            elif mode == 2:
                is_saving = not is_saving

    cv2.setMouseCallback(window_name, mouse_callback)
    print("\n--- Interactive Video Controls ---")
    print("  Spacebar: Play/Pause")
    print("  Mouse Click: Save (mode 1) or toggle recording (mode 2)")
    print("  Q: Quit")
    print("----------------------------------")

    while True:
        if not paused or current_frame is None:
            ret, frame = cap.read()
            if not ret:
                print("\n[Info] Reached end of video.")
                break
            current_frame = frame

            if mode == 2 and is_saving:
                save_frame(current_frame)

        if mode == 1 and save_single and current_frame is not None:
            save_frame(current_frame)
            save_single = False

        if current_frame is None:
            display_image = np.zeros((480, 640, 3), dtype=np.uint8)
            text = "Loading video..."
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (display_image.shape[1] - text_size[0]) // 2
            text_y = (display_image.shape[0] + text_size[1]) // 2
            cv2.putText(display_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            display_image = current_frame.copy()
            if mode == 2 and is_saving:
                cv2.circle(display_image, (30, 30), 20, (0, 0, 255), -1)
                cv2.putText(display_image, "REC", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if paused:
                cv2.putText(display_image, "PAUSED", (display_image.shape[1] - 150, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(window_name, display_image)

        key = cv2.waitKey(delay if not paused else 50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s') and mode == 1 and current_frame is not None:
            save_frame(current_frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nExtraction finished. {saved_count} images saved.")
    return True

def split_dataset_for_training(dataset_dir, ratios, class_names, image_formats, output_dir=None, exist_ok=False):
    """Split a dataset into train/val(/test) subsets with flexible output control.

    Args:
        dataset_dir (str): Source dataset root containing an ``images`` directory.
        ratios (dict[str, int]): Ratio mapping (values must sum to > 0).
        class_names: Mapping or iterable of class labels for ``data.yaml``.
        image_formats (Iterable[str]): Image extensions to scan for.
        output_dir (str | None): Destination dataset root. If ``None`` the source
            dataset is modified in place. When provided, files are copied to the
            destination, leaving the source untouched.
        exist_ok (bool): Overwrite the destination directory when ``output_dir``
            is provided and already exists.
    """

    source_root = Path(dataset_dir).resolve()
    target_root = Path(output_dir).resolve() if output_dir else source_root
    in_place = target_root == source_root

    if not source_root.exists():
        print(f"[Error] Source dataset path does not exist: {source_root}")
        return False

    if not in_place:
        if target_root.exists():
            if exist_ok:
                shutil.rmtree(target_root)
            else:
                print(f"[Error] Output directory already exists: {target_root}")
                return False
        target_root.mkdir(parents=True, exist_ok=True)

    # Locate image-label pairs under any directory named 'images'
    all_image_files = sorted([
        p for ext in image_formats
        for p in glob.glob(os.path.join(dataset_dir, '**', f'*.{ext}'), recursive=True)
    ])
    sep = os.path.sep
    image_paths = [path for path in all_image_files if f'{sep}images{sep}' in path]

    if not image_paths:
        print(f"[Error] No images found within any 'images' subdirectory in '{dataset_dir}'.")
        return False

    subsets = list(ratios.keys())
    example_subsets = ",".join(subsets)
    print("\nPlease choose the desired output directory structure:")
    print(f"1: {target_root}/images/{{{example_subsets}}}, {target_root}/labels/{{{example_subsets}}}")
    print(f"2: {target_root}/{{{example_subsets}}}/images, {target_root}/{{{example_subsets}}}/labels")
    choice = input("Enter your choice (1 or 2): ")
    while choice not in ['1', '2']:
        choice = input("Invalid input. Please enter 1 or 2: ")
    structure_type = int(choice)

    # Build list of image/label pairs that actually exist
    valid_pairs = [(img, get_label_path(img)) for img in image_paths if os.path.exists(get_label_path(img))]
    if not valid_pairs:
        print("[Warning] No valid image-label pairs found to split.")
        return False

    random.shuffle(valid_pairs)
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        print("[Error] Sum of ratios must be positive.")
        return False
    normalized_ratios = {k: v / total_ratio for k, v in ratios.items()}

    def contains_source_items(target_dir_path: Path, index: int) -> bool:
        try:
            target_resolved = target_dir_path.resolve()
        except FileNotFoundError:
            return False
        for pair in valid_pairs:
            src_path = Path(pair[index])
            try:
                if src_path.resolve().is_relative_to(target_resolved):
                    return True
            except FileNotFoundError:
                continue
        return False

    subset_destinations: dict[str, tuple[Path, Path]] = {}
    for subset in subsets:
        if structure_type == 1:
            img_dir = target_root / 'images' / subset
            lbl_dir = target_root / 'labels' / subset
        else:
            img_dir = target_root / subset / 'images'
            lbl_dir = target_root / subset / 'labels'

        # Avoid removing directories that contain the source files when operating in place
        if img_dir.exists() and not (in_place and contains_source_items(img_dir, 0)):
            shutil.rmtree(img_dir)
        if lbl_dir.exists() and not (in_place and contains_source_items(lbl_dir, 1)):
            shutil.rmtree(lbl_dir)

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        subset_destinations[subset] = (img_dir, lbl_dir)

    # Determine assignments per subset according to ratios
    assignments: dict[str, list[tuple[str, str]]] = {subset: [] for subset in subsets}
    start_index = 0
    num_files = len(valid_pairs)
    for i, (subset, ratio) in enumerate(normalized_ratios.items()):
        end_index = num_files if i == len(normalized_ratios) - 1 else start_index + int(num_files * ratio)
        assignments[subset] = valid_pairs[start_index:end_index]
        start_index = end_index

    for subset, pairs in assignments.items():
        dest_img_dir, dest_lbl_dir = subset_destinations[subset]
        print(f"Moving {len(pairs)} pairs to '{subset}'...")
        for img_path, lbl_path in tqdm(pairs, desc=f"Processing {subset} files"):
            src_img = Path(img_path)
            src_lbl = Path(lbl_path)
            dest_img = dest_img_dir / src_img.name
            dest_lbl = dest_lbl_dir / src_lbl.name

            if in_place:
                if src_img.resolve() != dest_img.resolve():
                    shutil.move(str(src_img), dest_img)
                if src_lbl.resolve() != dest_lbl.resolve():
                    shutil.move(str(src_lbl), dest_lbl)
            else:
                shutil.copy2(str(src_img), dest_img)
                shutil.copy2(str(src_lbl), dest_lbl)

    if in_place and structure_type == 2:
        for legacy_name in ('images', 'labels'):
            legacy_dir = source_root / legacy_name
            if legacy_dir.exists():
                try:
                    shutil.rmtree(legacy_dir)
                except OSError as exc:
                    print(f"[Warning] Failed to remove legacy directory '{legacy_dir}': {exc}")

    yaml_path = target_root / 'data.yaml'
    if structure_type == 1:
        train_path = os.path.join('images', 'train')
        val_path = os.path.join('images', 'val')
        test_path = os.path.join('images', 'test') if 'test' in subsets else None
    else:
        train_path = os.path.join('train', 'images')
        val_path = os.path.join('val', 'images')
        test_path = os.path.join('test', 'images') if 'test' in subsets else None

    if isinstance(class_names, dict):
        sorted_items = sorted(class_names.items())
        names_list = [str(name) for _, name in sorted_items]
    else:
        names_list = [str(name) for name in list(class_names)]

    data = {
        'path': str(target_root.resolve()),
        'train': train_path,
        'val': val_path,
        'names': names_list,
    }
    if test_path:
        data['test'] = test_path

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    if in_place:
        print(f"Dataset split complete in-place at: {source_root}")
    else:
        print(f"Dataset split complete. New dataset created at: {target_root}")
    return True

def merge_datasets(merge_config, output_dir, exist_ok=False):
    """Merge datasets using the configuration gathered from the CLI."""
    profiles: List[MergeDatasetProfile] = merge_config.get('profiles', [])
    target_structure: str = merge_config.get('target_structure', 'images_first')
    target_subsets: List[str] = merge_config.get('target_subsets', ['train', 'val'])
    ratio_plan: Dict[str, object] = merge_config.get('ratio_plan', {'mode': 'preserve'})

    if not profiles:
        print("[Error] No dataset profiles supplied for merge.")
        return False

    target_subsets = [subset for subset in target_subsets if subset in STANDARD_SUBSETS]
    if not target_subsets:
        print("[Error] No valid target subsets specified.")
        return False

    if target_structure not in {'images_first', 'subset_first'}:
        print(f"[Error] Unknown target structure '{target_structure}'.")
        return False

    output_path = Path(output_dir)
    if output_path.exists():
        if not exist_ok:
            print(f"[Error] Output directory already exists: {output_dir}")
            return False
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    def assign_by_ratio(pairs: List[Tuple[str, str]], ratio_map: Dict[str, float]) -> Dict[str, List[Tuple[str, str]]]:
        assignments = {subset: [] for subset in target_subsets}
        if not pairs:
            return assignments

        pool = pairs[:]
        random.shuffle(pool)
        total = len(pool)
        floored = []
        for subset in target_subsets:
            value = max(0.0, ratio_map.get(subset, 0.0))
            floored.append(int(total * value))

        distributed = sum(floored)
        if distributed > total:
            overflow = distributed - total
            for idx in range(len(floored)):
                if overflow <= 0:
                    break
                reduction = min(overflow, floored[idx])
                floored[idx] -= reduction
                overflow -= reduction

        remainder = total - sum(floored)
        index = 0
        for idx, subset in enumerate(target_subsets):
            count = floored[idx]
            if idx == len(target_subsets) - 1:
                count += remainder
            assignments[subset] = pool[index:index + count]
            index += count

        if index < total:
            assignments[target_subsets[-1]].extend(pool[index:])

        return assignments

    final_assignments: Dict[str, List[Tuple[str, str]]] = {subset: [] for subset in target_subsets}

    for profile in profiles:
        if ratio_plan.get('mode') == 'preserve':
            for subset in target_subsets:
                final_assignments[subset].extend(profile.subset_pairs.get(subset, []))
        else:
            pool: List[Tuple[str, str]] = []
            for subset_pairs in profile.subset_pairs.values():
                pool.extend(subset_pairs)
            pool.extend(profile.unassigned_pairs)

            if not pool:
                continue

            if ratio_plan.get('mode') == 'uniform':
                ratio_map = ratio_plan['ratios']
            elif ratio_plan.get('mode') == 'per_dataset':
                ratio_map = ratio_plan['ratios'].get(profile.path, {})
            else:
                ratio_map = {}

            ratio_map = {subset: ratio_map.get(subset, 0.0) for subset in target_subsets}
            assignments = assign_by_ratio(pool, ratio_map)
            for subset in target_subsets:
                final_assignments[subset].extend(assignments[subset])

    def ensure_destination_dirs(subset_name: str) -> Tuple[Path, Path]:
        if target_structure == 'images_first':
            img_dir = output_path / 'images' / subset_name
            lbl_dir = output_path / 'labels' / subset_name
        else:
            img_dir = output_path / subset_name / 'images'
            lbl_dir = output_path / subset_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        return img_dir, lbl_dir

    saved_counts: Dict[str, int] = {}
    for subset in target_subsets:
        img_dir, lbl_dir = ensure_destination_dirs(subset)
        counter = 0
        pairs = final_assignments.get(subset, [])
        if not pairs:
            saved_counts[subset] = 0
            continue

        for img_path, lbl_path in tqdm(pairs, desc=f"Copying {subset} pairs"):
            img_suffix = Path(img_path).suffix
            base_name = f"{counter:06d}"
            counter += 1
            shutil.copy2(img_path, img_dir / f"{base_name}{img_suffix}")
            shutil.copy2(lbl_path, lbl_dir / f"{base_name}.txt")

        saved_counts[subset] = counter

    summary = ", ".join(f"{subset}: {count}" for subset, count in saved_counts.items())
    print(f"\nMerge complete. Saved pairs per split -> {summary}")
    return True

def get_all_image_data(source_dir, image_formats):
    """
    Finds all images in a dataset, supporting both 'images/train' and 'train/images' structures.
    Returns a list of tuples: (image_path, label_path_or_None).
    """
    source_path = Path(source_dir)
    image_paths = []
    for fmt in image_formats:
        image_paths.extend(source_path.glob(f'**/*.{fmt}'))
    # This filter is the key to supporting both structures robustly.
    image_paths = [p for p in image_paths if 'images' in p.parts]
    if not image_paths: return []
    all_image_data = []
    for img_path in sorted(image_paths):
        label_path = get_label_path(str(img_path))
        if os.path.exists(label_path):
            all_image_data.append((str(img_path), label_path))
        else:
            all_image_data.append((str(img_path), None))
    return all_image_data

def sample_dataset(source_dir, output_dir, sample_ratio, image_formats, exist_ok=False, method='random'):
    """Creates a new, smaller dataset by sampling from a source dataset."""
    if os.path.exists(output_dir) and not exist_ok:
        print(f"[Error] Output directory '{output_dir}' already exists.")
        return
    if os.path.exists(output_dir): shutil.rmtree(output_dir)

    all_image_data = get_all_image_data(source_dir, image_formats)
    if not all_image_data:
        print(f"[Error] No valid images found in {source_dir}.")
        return

    num_samples = max(1, int(len(all_image_data) * sample_ratio))
    print(f"Sampling {num_samples} items ({sample_ratio*100:.1f}% of {len(all_image_data)} total images).")

    if method == 'random':
        sampled_data = random.sample(all_image_data, num_samples)
    elif method == 'uniform':
        indices = np.round(np.linspace(0, len(all_image_data) - 1, num_samples)).astype(int)
        sampled_data = [all_image_data[i] for i in np.unique(indices)]
    else:
        print(f"[Error] Unknown sampling method: {method}")
        return

    for img_path_str, label_path_str in tqdm(sampled_data, desc="Copying sampled data"):
        img_path = Path(img_path_str)
        try:
            parts = img_path.parts
            images_index = parts.index('images')
            relative_structure = Path(*parts[images_index:])
        except ValueError:
            relative_structure = Path('images') / img_path.name
        
        output_img_path = Path(output_dir) / relative_structure
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, output_img_path)

        if label_path_str:
            label_path = Path(label_path_str)
            label_relative_structure = Path(str(relative_structure).replace('images', 'labels', 1))
            output_label_path = Path(output_dir) / label_relative_structure
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, output_label_path)
    
    print("\nDataset sampling complete.")
