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
        from rosbag2_interfaces.srv import Pause, Resume, TogglePaused

        self.rclpy = rclpy
        if not self.rclpy.ok():
            self.rclpy.init()

        self.node = self.rclpy.create_node('yolo_toolkit_player_controller_node')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.bag_process = None
        self.is_paused = True

        # Store service types as instance attributes to be accessible by other methods
        self.PauseSrv = Pause
        self.TogglePausedSrv = TogglePaused

        # Create a subscriber to receive the images
        self.subscription = self.node.create_subscription(
            Image, image_topic, self._image_callback, 10)

        # Create clients to control the player node
        self.toggle_client = self.node.create_client(self.TogglePausedSrv, '/rosbag2_player/toggle_paused')
        self.pause_client = self.node.create_client(self.PauseSrv, '/rosbag2_player/pause')
        
        # Start the ROS2 node spinning in a separate thread
        self.ros_thread = threading.Thread(target=self.rclpy.spin, args=(self.node,), daemon=True)
        self.ros_thread.start()
        print("ROS2 subscriber and service client node started.")

    def _image_callback(self, msg):
        with self.frame_lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def play_bag(self, bag_path):
        command = ['ros2', 'bag', 'play', bag_path]
        print(f"Executing command: {' '.join(command)}")
        try:
            self.bag_process = subprocess.Popen(command)
            if not self.pause_client.wait_for_service(timeout_sec=5.0):
                print("[Error] Could not connect to /rosbag2_player/pause service.")
                self.cleanup()
                return False
            
            # Immediately pause it via a service call using the pre-made client
            self.pause_client.call_async(self.PauseSrv.Request())
            print("Player started and immediately paused via service call.")
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
        # (The non-interactive code block is unchanged and remains here)
        print("Running non-interactive extraction...")
        # ...
        return True

    # --- Interactive extraction (modes 1 and 2) ---
    player = RosBagPlayer(image_topic)
    if not player.play_bag(rosbag_dir):
        player.cleanup()
        return False

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
    
    print("Waiting for the first image from the bag...")
    while player.get_frame() is None:
        if player.bag_process and player.bag_process.poll() is not None:
            print("[Error] ros2 bag play process terminated unexpectedly.")
            player.cleanup()
            cv2.destroyAllWindows()
            return False
        time.sleep(0.1)
    
    print("First frame received. Player is ready (paused).")
    
    while True:
        if player.bag_process and player.bag_process.poll() is not None:
            print("\nROS2 bag play process has ended.")
            break
            
        frame = player.get_frame()
        if frame is None:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            continue

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

# ... (rest of the file: split_dataset_for_training, merge_datasets, etc. remains unchanged) ...