import os
import glob
import shutil
import random
from pathlib import Path
import yaml
import time
import sys
from tqdm import tqdm
import numpy as np
import subprocess
import threading

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

def split_dataset_for_training(dataset_dir, ratios, class_names, image_formats):
    """Splits a dataset into multiple subsets based on given ratios, supporting flexible structures."""
    # Robustly find all images within any 'images' subdirectory
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
    print(f"1: {dataset_dir}/images/{{{example_subsets}}}, {dataset_dir}/labels/{{{example_subsets}}}")
    print(f"2: {dataset_dir}/{{{example_subsets}}}/images, {dataset_dir}/{{{example_subsets}}}/labels")
    choice = input("Enter your choice (1 or 2): ")
    while choice not in ['1', '2']:
        choice = input("Invalid input. Please enter 1 or 2: ")
    structure_type = int(choice)
    
    # Create output directories based on chosen structure
    if structure_type == 1:
        for sub in subsets:
            os.makedirs(os.path.join(dataset_dir, 'images', sub), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, 'labels', sub), exist_ok=True)
    else:  # structure_type == 2
        for sub in subsets:
            os.makedirs(os.path.join(dataset_dir, sub, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, sub, 'labels'), exist_ok=True)

    valid_pairs = [p for p in image_paths if os.path.exists(get_label_path(p))]
    if not valid_pairs:
        print("[Warning] No valid image-label pairs found to split.")
        return False
    random.shuffle(valid_pairs)
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        print("[Error] Sum of ratios must be positive.")
        return False
    normalized_ratios = {k: v / total_ratio for k, v in ratios.items()}

    def move_pair(file_path, subset):
        try:
            label_path = get_label_path(file_path)
            if structure_type == 1:
                img_dest = os.path.join(dataset_dir, 'images', subset, os.path.basename(file_path))
                lbl_dest = os.path.join(dataset_dir, 'labels', subset, os.path.basename(label_path))
            else:  # structure_type == 2
                img_dest = os.path.join(dataset_dir, subset, 'images', os.path.basename(file_path))
                lbl_dest = os.path.join(dataset_dir, subset, 'labels', os.path.basename(label_path))
            
            shutil.move(file_path, img_dest)
            shutil.move(label_path, lbl_dest)
        except FileNotFoundError:
            print(f"[Warning] Could not find image or label for: {os.path.basename(file_path)}")

    start_index = 0
    num_files = len(valid_pairs)
    for i, (subset, ratio) in enumerate(normalized_ratios.items()):
        end_index = num_files if i == len(normalized_ratios) - 1 else start_index + int(num_files * ratio)
        
        files_to_move = valid_pairs[start_index:end_index]
        print(f"Moving {len(files_to_move)} pairs to '{subset}'...")
        for p in tqdm(files_to_move, desc=f"Moving {subset} files"):
            move_pair(p, subset)
        start_index = end_index

    # Create data.yaml based on chosen structure
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    if structure_type == 1:
        train_path = os.path.join('images', 'train')
        val_path = os.path.join('images', 'val')
        test_path = os.path.join('images', 'test') if 'test' in subsets else None
    else:  # structure_type == 2
        train_path = os.path.join('train', 'images')
        val_path = os.path.join('val', 'images')
        test_path = os.path.join('test', 'images') if 'test' in subsets else None

    data = {
        'path': os.path.abspath(dataset_dir), 'train': train_path, 'val': val_path,
        'names': [n for _, n in sorted(class_names.items())]
    }
    if test_path: data['test'] = test_path
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    print("Dataset split complete.")
    return True

def merge_datasets(input_dirs, output_dir, image_formats, exist_ok=False, strategy='flatten', base_dataset=None):
    """Merges multiple datasets using either a 'flatten' or 'structured' strategy."""
    if os.path.exists(output_dir) and not exist_ok:
        print(f"[Error] Output directory already exists: {output_dir}")
        return False
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    sep = os.path.sep

    if strategy == 'flatten':
        print("\nRunning Flatten Merge...")
        out_img = os.path.join(output_dir, 'images'); os.makedirs(out_img)
        out_lbl = os.path.join(output_dir, 'labels'); os.makedirs(out_lbl)
        
        all_files_unfiltered = [p for d in input_dirs for fmt in image_formats for p in glob.glob(os.path.join(d, '**', f'*.{fmt}'), recursive=True)]
        all_images = sorted([p for p in all_files_unfiltered if f'{sep}images{sep}' in p])
        
        c = 0
        for img_path in tqdm(all_images, desc="Merging and flattening"):
            lbl_path = get_label_path(img_path)
            if os.path.exists(lbl_path):
                ext = os.path.splitext(img_path)[1]
                new_base = f"{c:06d}"
                shutil.copy2(img_path, os.path.join(out_img, new_base + ext))
                shutil.copy2(lbl_path, os.path.join(out_lbl, new_base + '.txt'))
                c += 1
        print(f"\nFlatten merge complete. {c} image-label pairs saved.")
        return True

    elif strategy == 'structured':
        if not base_dataset or base_dataset not in input_dirs:
            print("[Error] A valid base dataset must be selected for structured merge.")
            return False
        
        print(f"\nRunning Structured Merge based on '{os.path.basename(base_dataset)}'...")
        base_path = Path(base_dataset)
        
        unfiltered_base_images = [p for fmt in image_formats for p in base_path.glob(f'**/*.{fmt}')]
        base_images = [p for p in unfiltered_base_images if 'images' in p.parts]
        
        rel_img_subdirs = sorted(list(set([p.relative_to(base_path).parent for p in base_images if 'images' in p.parts])))

        if not rel_img_subdirs:
            print(f"[Error] No image subdirectories found in the base dataset: {base_dataset}")
            return False

        print(f"Base structure detected: {[str(p) for p in rel_img_subdirs]}")
        total_saved_pairs = 0
        for rel_img_subdir in rel_img_subdirs:
            out_img_subdir = Path(output_dir) / rel_img_subdir
            rel_lbl_subdir = Path(str(rel_img_subdir).replace('images', 'labels', 1))
            out_lbl_subdir = Path(output_dir) / rel_lbl_subdir
            out_img_subdir.mkdir(parents=True, exist_ok=True)
            out_lbl_subdir.mkdir(parents=True, exist_ok=True)
            file_counter = 0
            for source_dir in input_dirs:
                current_scan_dir = Path(source_dir) / rel_img_subdir
                if not current_scan_dir.is_dir(): continue
                # Recursively search within the subdirectory as well for robustness
                images_in_subdir = sorted([p for fmt in image_formats for p in current_scan_dir.glob(f'**/*.{fmt}')])
                for img_path in images_in_subdir:
                    lbl_path = get_label_path(str(img_path))
                    if os.path.exists(lbl_path):
                        ext = img_path.suffix
                        new_base = f"{file_counter:06d}"
                        shutil.copy2(str(img_path), out_img_subdir / (new_base + ext))
                        shutil.copy2(lbl_path, out_lbl_subdir / (new_base + '.txt'))
                        file_counter += 1
            print(f" - Merged {file_counter} pairs into '{rel_img_subdir}'")
            total_saved_pairs += file_counter
        print(f"\nStructured merge complete. {total_saved_pairs} total image-label pairs saved.")
        return True
    
    else:
        print(f"[Error] Unknown merge strategy: '{strategy}'")
        return False

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
