import cv2
import os
import glob
import shutil
import yaml
import argparse

# ==============================================================================
# Initiation Settings
# ==============================================================================
# These global variables define the default parameters for the visualizer script.
# They are used as fallbacks if corresponding command-line arguments are not
# provided, allowing the script to run with a predefined configuration.

# Default relative path to the dataset directory to be processed.
INIT_DATASET_DIR = None

# Default execution mode ('visualize' or 'remove_noise').
INIT_MODE = 'visualize'

# Default relative path for saving the cleaned dataset in 'remove_noise' mode.
INIT_CLEANED_DIR = None
# ==============================================================================

class Visualizer:
    """
    A class to visualize, review, and clean a labeled dataset.
    """
    def __init__(self, dataset_dir, mode, cleaned_dir, config):
        self.dataset_dir, self.mode, self.cleaned_dir, self.config = dataset_dir, mode, cleaned_dir, config
        self.classes = config['model_configurations']['classes']
        self.colors = {c:((c*40+50)%256, (c*80+100)%256, (c*120+150)%256) for c in self.classes.keys()}
        self.is_paused, self.img_index = True, 0
        self.review_files, self.noise_files = set(), set()
        self.window_name = f"Label Visualizer"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self.image_paths, self.is_split = self._load_image_paths()

    def _load_image_paths(self):
        """Automatically detects dataset structure (split/unified) and loads all image paths."""
        img_fmts = self.config['workflow_parameters']['image_format'].split(',')
        is_split = os.path.isdir(os.path.join(self.dataset_dir, 'images', 'train'))
        print(f"  - Dataset Structure: {'Split' if is_split else 'Unified'}")
        if is_split:
            base_dirs = [os.path.join(self.dataset_dir, 'images', sub) for sub in ['train', 'val']]
        else:
            base_dirs = [os.path.join(self.dataset_dir, 'images')]
        paths = [p for d in base_dirs for fmt in img_fmts for p in glob.glob(os.path.join(d, f'*.{fmt}'))]
        return sorted(paths), is_split

    def _mouse_callback(self, event, x, y, flags, param):
        """Handles mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            img_name = os.path.basename(self.image_paths[self.img_index])
            target_set, msg = (self.review_files, "review list") if self.mode == 'visualize' else (self.noise_files, "removal list")
            action = "added" if img_name not in target_set else "removed"
            (target_set.add if action == "added" else target_set.remove)(img_name)
            print(f"  -> '{img_name}' {action} to {msg} (Total: {len(target_set)})")

    def _create_cleaned_dataset(self):
        """Saves a new dataset excluding frames marked as noise."""
        print(f"\n[Noise Removal] Saving cleaned dataset to '{self.cleaned_dir}'.")
        if os.path.exists(self.cleaned_dir):
            shutil.rmtree(self.cleaned_dir); print(f"  - Warning: Deleting existing '{self.cleaned_dir}' and creating a new one.")
        cleaned_images_dir = os.path.join(self.cleaned_dir, 'images')
        cleaned_labels_dir = os.path.join(self.cleaned_dir, 'labels')
        os.makedirs(cleaned_images_dir); os.makedirs(cleaned_labels_dir)
        copied_count = 0
        for img_path in self.image_paths:
            img_filename = os.path.basename(img_path)
            if img_filename not in self.noise_files:
                sub_dir = os.path.basename(os.path.dirname(img_path)) if self.is_split else ''
                label_filename = os.path.splitext(img_filename)[0] + '.txt'
                label_path = os.path.join(self.dataset_dir, 'labels', sub_dir, label_filename)
                shutil.copy2(img_path, os.path.join(cleaned_images_dir, img_filename))
                if os.path.exists(label_path): shutil.copy2(label_path, os.path.join(cleaned_labels_dir, label_filename))
                copied_count += 1
        print(f"  - Excluded {len(self.noise_files)} noise frames from {len(self.image_paths)} total, copied {copied_count} files.")

    def run(self):
        """Runs the main visualization loop."""
        if not self.image_paths: print(f"Error: No images found in '{self.dataset_dir}'."); return
        print("\nControls: [Space]: Autoplay/Pause | [d]: Next | [a]: Previous | [Click]: Select/Deselect | [q]: Quit")
        while 0 <= self.img_index < len(self.image_paths):
            img_path = self.image_paths[self.img_index]; img_name = os.path.basename(img_path)
            sub_dir = os.path.basename(os.path.dirname(img_path)) if self.is_split else ''
            label_path = os.path.join(self.dataset_dir, 'labels', sub_dir, os.path.splitext(img_name)[0] + '.txt')
            img = cv2.imread(img_path)
            if img is None: self.img_index += 1; continue
            
            cv2.setWindowTitle(self.window_name, f"Label Visualizer - {self.mode.upper()} MODE | {img_name}")
            
            h, w = img.shape[:2]
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        cid, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
                        x1,y1,x2,y2 = int((xc-bw/2)*w), int((yc-bh/2)*h), int((xc+bw/2)*w), int((yc+bh/2)*h)
                        color = self.colors.get(cid, (255,255,255))
                        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(img, f'{cid}:{self.classes.get(cid, "Unknown")}', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if self.mode == 'visualize' and img_name in self.review_files:
                cv2.putText(img, "REVIEW", (20,50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 3)
            elif self.mode == 'remove_noise' and img_name in self.noise_files:
                cv2.putText(img, "TO BE REMOVED", (20,50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (50,50,200), 3)
            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(100 if not self.is_paused else -1) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '): self.is_paused = not self.is_paused
            if self.is_paused:
                if key == ord('d'): self.img_index = min(self.img_index + 1, len(self.image_paths) - 1)
                elif key == ord('a'): self.img_index = max(self.img_index - 1, 0)
            else: self.img_index += 1
        if self.mode == 'visualize' and self.review_files:
            review_list_path = os.path.join(self.dataset_dir, 'review_list.txt')
            with open(review_list_path, 'w') as f: f.write('\n'.join(sorted(list(self.review_files))))
            print(f"\nReview list saved to '{review_list_path}'.")
        elif self.mode == 'remove_noise' and self.noise_files: self._create_cleaned_dataset()
        cv2.destroyAllWindows(); print("\nVisualizer tool closed.")

def main(config, args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dataset_rel = args.dataset or INIT_DATASET_DIR or config['dataset_paths'].get('reviewed_dataset', 'datasets/default_review')
    dataset_dir = os.path.join(project_root, dataset_rel)
    mode = args.mode or INIT_MODE
    cleaned_dir_rel = args.output_dir or INIT_CLEANED_DIR or config['dataset_paths'].get('reviewed_dataset', 'datasets/default_cleaned')
    cleaned_dir = os.path.join(project_root, cleaned_dir_rel)
    
    print("\n" + "="*50); print("Starting Label Visualizer and Cleaner."); print("="*50)
    print(f"  - Target Dataset: {dataset_dir}"); print(f"  - Mode: {mode}")
    if mode == 'remove_noise': print(f"  - Cleaned Data Output Path: {cleaned_dir}"); print("="*50)
    
    visualizer = Visualizer(dataset_dir, mode, cleaned_dir, config); visualizer.run()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f: config = yaml.safe_load(f)
    except FileNotFoundError: print("Error: _config.yaml not found."); exit()
    parser = argparse.ArgumentParser(description="Visualizes labeled datasets or removes noise.")
    parser.add_argument('--dataset', type=str, default=None, help="Relative path to the dataset to process.")
    parser.add_argument('--mode', type=str, default=None, choices=['visualize', 'remove_noise'], help="Select execution mode.")
    parser.add_argument('--output_dir', type=str, default=None, help="Path to save the cleaned dataset in 'remove_noise' mode.")
    args = parser.parse_args()
    main(config, args)