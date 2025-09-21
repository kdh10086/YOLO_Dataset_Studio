
import os
import glob
import shutil
import random
import yaml
import time
import sys
from tqdm import tqdm

from toolkit.utils import get_label_path

def _get_topic_type(bag_dir, topic_name):
    metadata_path = os.path.join(bag_dir, 'metadata.yaml')
    if not os.path.exists(metadata_path): return None
    with open(metadata_path, 'r') as f: metadata = yaml.safe_load(f)['rosbag2_bagfile_information']
    for topic_info in metadata['topics_with_message_count']:
        if topic_info['topic_metadata']['name'] == topic_name: return topic_info['topic_metadata']['type']
    return None

def extract_images_from_rosbag(rosbag_dir, output_dir, image_topic, image_formats, mode=0):
    """
    Extracts images from a ROS2 bag file.
    Mode 0: Non-interactive, extracts all images.
    Modes 1 & 2: Interactive GUI modes.
    """
    try:
        import cv2
        from cv_bridge import CvBridge
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        print("[Error] ROS2/OpenCV libraries not found. Cannot proceed with extraction.")
        return False

    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    start_index = 0
    existing = [int(os.path.splitext(f)[0]) for f in os.listdir(images_dir) if f.split('.')[-1] in image_formats and f.split('.')[0].isdigit()]
    if existing:
        start_index = max(existing) + 1

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
    saved_count = 0

    # --- Non-interactive extraction (mode 0) ---
    if mode == 0:
        print("Extracting all images from topic. This may take a while...")
        for topic, data, t in tqdm(reader, desc="Extracting from Bag"):
            if topic == image_topic:
                try:
                    cv_image = bridge.imgmsg_to_cv2(deserialize_message(data, msg_type), "bgr8")
                    fname = f"{start_index:06d}.{image_formats[0]}"
                    cv2.imwrite(os.path.join(images_dir, fname), cv_image)
                    saved_count += 1
                    start_index += 1
                except Exception as e:
                    print(f"\n[Warning] Could not process a message: {e}")
        print(f"\nExtraction finished. {saved_count} images saved.")
        return True

    # --- Interactive extraction (modes 1 and 2) ---
    is_paused, is_saving, save_single, cv_image = True, False, False, None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal save_single, is_saving
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == 1: # Single-save mode
                save_single = True
            elif mode == 2: # Range-save mode
                is_saving = not is_saving

    cv2.namedWindow("ROS2 Bag Player")
    cv2.setMouseCallback("ROS2 Bag Player", mouse_callback)
    print("\n--- Interactive Player Controls ---")
    print("  Spacebar: Play/Pause")
    print("  Mouse Click: Save image (single or toggle range based on mode)")
    print("  Q: Quit")
    print("---------------------------------")

    while reader.has_next():
        if not is_paused or cv_image is None:
            try:
                topic, data, t = reader.read_next()
                if topic == image_topic:
                    cv_image = bridge.imgmsg_to_cv2(deserialize_message(data, msg_type), "bgr8")
            except Exception as e:
                print(f"\n[Warning] End of bag or error reading message: {e}")
                break
        
        if cv_image is None:
            continue

        display_image = cv_image.copy()
        
        # Visual feedback for saving state
        if is_saving and mode == 2:
            cv2.circle(display_image, (30, 30), 20, (0, 0, 255), -1)
            cv2.putText(display_image, "REC", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if (mode == 1 and save_single) or (mode == 2 and is_saving):
            fname = f"{start_index:06d}.{image_formats[0]}"
            cv2.imwrite(os.path.join(images_dir, fname), cv_image)
            saved_count += 1
            start_index += 1
            save_single = False # Reset after single save

        cv2.imshow("ROS2 Bag Player", display_image)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            is_paused = not is_paused

    cv2.destroyAllWindows()
    print(f"\nExtraction finished. {saved_count} images saved.")
    return True

# ==============================================================================
# DATASET MANIPULATION
# ==============================================================================

def split_dataset_for_training(dataset_dir, train_ratio, class_names, image_formats):
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"[Error] 'images' or 'labels' directory not found in '{dataset_dir}'.")
        return False

    # Create train/val subdirectories
    for sub in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, sub), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, sub), exist_ok=True)

    # Find all image files and ensure they have a corresponding label file
    image_paths = [p for fmt in image_formats for p in glob.glob(os.path.join(images_dir, f'*.{fmt}'))]
    valid_pairs = [p for p in image_paths if os.path.exists(get_label_path(p))]
    
    if not valid_pairs:
        print("[Warning] No valid image-label pairs found to split.")
        return False

    random.shuffle(valid_pairs)
    split_point = int(len(valid_pairs) * train_ratio)
    train_files = valid_pairs[:split_point]
    val_files = valid_pairs[split_point:]

    # Move files in a clear, atomic way
    def move_pair(file_path, subset):
        try:
            # Move image
            shutil.move(file_path, os.path.join(images_dir, subset, os.path.basename(file_path)))
            # Move label
            label_path = get_label_path(file_path)
            shutil.move(label_path, os.path.join(labels_dir, subset, os.path.basename(label_path)))
        except FileNotFoundError:
            print(f"[Warning] Could not find image or label for: {os.path.basename(file_path)}")

    print(f"Moving {len(train_files)} pairs to 'train'...")
    for p in tqdm(train_files, desc="Moving train files"):
        move_pair(p, 'train')

    print(f"Moving {len(val_files)} pairs to 'val'...")
    for p in tqdm(val_files, desc="Moving val files"):
        move_pair(p, 'val')

    # Create data.yaml
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    yaml_content = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': [n for _, n in sorted(class_names.items())]
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print("Dataset split complete.")
    return True

def merge_datasets(input_dirs, output_dir, image_formats, exist_ok=False):
    if os.path.exists(output_dir) and not exist_ok: print(f"[Error] Output dir exists: {output_dir}"); return False
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    out_img = os.path.join(output_dir,'images'); out_lbl = os.path.join(output_dir,'labels'); os.makedirs(out_img); os.makedirs(out_lbl)
    all_images = [p for d in input_dirs for fmt in image_formats for p in glob.glob(os.path.join(d,'**',f'*.{fmt}'),recursive=True)]
    c=0
    for img_path in tqdm(all_images, desc="Merging"): 
        lbl_path = get_label_path(img_path)
        if os.path.exists(lbl_path):
            ext=os.path.splitext(img_path)[1]; new_base=f"{c:06d}"
            shutil.copy2(img_path, os.path.join(out_img, new_base+ext)); shutil.copy2(lbl_path, os.path.join(out_lbl, new_base+'.txt')); c+=1
    print(f"Merge complete. {c} pairs saved."); return True

def sample_dataset(source_dir, output_dir, sample_ratio, image_formats, exist_ok=False):
    if os.path.exists(output_dir) and not exist_ok: print(f"[Error] Output dir exists: {output_dir}"); return False
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    out_img = os.path.join(output_dir,'images'); out_lbl = os.path.join(output_dir,'labels'); os.makedirs(out_img); os.makedirs(out_lbl)
    all_images = [p for fmt in image_formats for p in glob.glob(os.path.join(source_dir,'**',f'*.{fmt}'),recursive=True)]
    pairs = [{'image': p, 'label': get_label_path(p)} for p in all_images if os.path.exists(get_label_path(p))]
    random.shuffle(pairs); sampled = pairs[:int(len(pairs)*sample_ratio)]
    for pair in tqdm(sampled, desc="Sampling"): shutil.copy2(pair['image'], out_img); shutil.copy2(pair['label'], out_lbl)
    print(f"Sampling complete. {len(sampled)} pairs saved."); return True
