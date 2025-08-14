import cv2
import os
import yaml
import argparse
import time
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge

# ==============================================================================
# Initiation Settings
# ==============================================================================
# Sets the default ROS2 bag directory path.
# This is used if the '--bag' command-line argument is not provided.
# If this is also None, the script uses the value from '_config.yaml'.
# Example: '/path/to/your/rosbag_directory'
INIT_BAG_DIR = None

# Sets the default output directory path, relative to the project root.
# This is used if the '--output' command-line argument is not provided.
# If this is also None, the script uses the value from '_config.yaml'.
# Example: 'output/session1'
INIT_OUTPUT_DIR = None

# Sets the default extraction mode.
# This is used if the '--mode' command-line argument is not provided.
# Mode 1: Single Frame Save. Click to save the currently displayed frame.
# Mode 2: Frame Range Save. Click to start/stop saving a sequence of frames.
INIT_MODE = None
# ==============================================================================

def get_topic_type(bag_dir, topic_name):
    """
    Finds and returns the message type for a specified topic from the metadata.yaml of a ROS2 Bag.
    """
    metadata_path = os.path.join(bag_dir, 'metadata.yaml')
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at path: {metadata_path}")
        return None
    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)['rosbag2_bagfile_information']
        for topic_info in metadata['topics_with_message_count']:
            if topic_info['topic_metadata']['name'] == topic_name:
                return topic_info['topic_metadata']['type']
    except Exception as e:
        print(f"Error processing metadata file: {e}")
    return None

def extract_frames(config, args):
    """
    Main function to read a ROS2 bag, display images, and save them based on user input.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get settings from the _config.yaml structure
    bag_dir = args.bag if args.bag is not None else INIT_BAG_DIR if INIT_BAG_DIR is not None else config['dataset_paths']['ros2bag_directory']
    output_dir_relative = args.output if args.output is not None else INIT_OUTPUT_DIR if INIT_OUTPUT_DIR is not None else config['dataset_paths']['extracted_images_dir']
    output_dir = os.path.join(project_root, output_dir_relative)
    mode = args.mode if args.mode is not None else INIT_MODE if INIT_MODE is not None else 1
    image_topic = config['workflow_parameters']['ros2_image_topic']
    
    print("\n" + "="*50); print("Starting ROS2 Bag Image Extraction."); print("="*50)
    print(f"  - Bag File Path: {bag_dir}")
    print(f"  - Output Directory: {output_dir}")
    print(f"  - Image Topic: {image_topic}")
    print(f"  - Extraction Mode: {'Single Frame Save' if mode == 1 else 'Frame Range Save'}")
    print("="*50)

    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    image_formats = config['workflow_parameters']['image_format'].split(',')
    existing_files = os.listdir(images_dir)
    if not existing_files:
        start_index = 0
    else:
        indices = [int(os.path.splitext(f)[0]) for f in existing_files if f.split('.')[-1] in image_formats and f.split('.')[0].isdigit()]
        start_index = max(indices) + 1 if indices else 0

    storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error: Failed to open ROS Bag: {e}"); return

    topic_type_str = get_topic_type(bag_dir, image_topic)
    if not topic_type_str:
        print(f"Error: Topic '{image_topic}' not found in the Bag file."); return
        
    msg_type = get_message(topic_type_str)
    bridge = CvBridge()
    
    is_paused = True
    is_saving = False
    save_single_frame_flag = False
    saved_count = 0
    cv_image = None
    last_print_time = 0  # Stores the timestamp of the last console print

    def mouse_callback(event, x, y, flags, param):
        nonlocal save_single_frame_flag, is_saving
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == 1:
                save_single_frame_flag = True
                print(f"-> Single frame save command received!")
            elif mode == 2:
                is_saving = not is_saving
                status_text = "Started" if is_saving else "Stopped"
                print(f"-> Range saving {status_text}.")

    cv2.namedWindow("ROS2 Bag Player")
    cv2.setMouseCallback("ROS2 Bag Player", mouse_callback)

    print("\nControls: [Space]: Play/Pause | [Left Mouse Click]: Save Action | [Q]: Quit")
    while reader.has_next():
        frame_updated = False # Flag to indicate a new frame has been loaded
        if not is_paused or cv_image is None:
            try:
                (topic, data, t) = reader.read_next()
                if topic == image_topic:
                    msg = deserialize_message(data, msg_type)
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    frame_updated = True
            except Exception as e:
                print(f"\nReached end of Bag file or a read error occurred: {e}"); break
        
        if cv_image is None: continue
        
        display_image = cv_image.copy()
        
        status_text = ""
        text_color = (0, 255, 0)
        if is_paused:
            status_text = "[PAUSED] "
            text_color = (0, 255, 255)
            if mode == 2 and is_saving:
                status_text += "[RECORDING ARMED]"
        else:
            if mode == 2 and is_saving:
                status_text = f"[RECORDING] Frames: {saved_count}"
                text_color = (0, 0, 255)
            else:
                status_text = "[PLAYING]"
        cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.imshow("ROS2 Bag Player", display_image)
        
        should_save = (mode == 1 and save_single_frame_flag) or \
                      (mode == 2 and is_saving and frame_updated)
        
        if should_save:
            filename = f"{start_index:06d}.{image_formats[0]}"
            filepath = os.path.join(images_dir, filename)
            cv2.imwrite(filepath, cv_image)
            
            saved_count += 1
            start_index += 1

            if mode == 1:
                print(f"  - Saved: {filename}")
                save_single_frame_flag = False
            elif mode == 2:
                current_time = time.time()
                if last_print_time == 0 or current_time - last_print_time >= 1.0:
                    print(f"  - Saving... (frame: {filename})")
                    last_print_time = current_time

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): is_paused = not is_paused

    cv2.destroyAllWindows()
    print(f"\nImage extraction finished. Total {saved_count} images saved.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f: config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: _config.yaml not found. Check if the file exists in the project root."); exit()

    parser = argparse.ArgumentParser(description="Extracts images from a ROS2 Bag file.")
    parser.add_argument('--bag', type=str, default=None, help="Absolute path to the ROS2 Bag directory.")
    parser.add_argument('--output', type=str, default=None, help="Relative path to the output directory for saving images.")
    parser.add_argument('--mode', type=int, default=None, choices=[1, 2], help="Extraction mode (1: Single, 2: Range).")
    args = parser.parse_args()
    extract_frames(config, args)