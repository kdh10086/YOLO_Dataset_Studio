import cv2
import os
import glob
import yaml
import argparse

class DatasetVisualizer:
    def __init__(self, config, args):
        """Initialize the visualizer, setting up paths, configurations, and review lists."""
        self.config = config
        self.args = args
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Setup dataset paths
        default_dataset = self.config['dataset_paths']['initial_labeled_dataset']
        dataset_rel = self.args.dataset or default_dataset
        self.dataset_dir = os.path.join(self.project_root, dataset_rel)
        
        # Setup images and labels
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.labels_dir = os.path.join(self.dataset_dir, 'labels')
        self.image_paths = sorted([p for ext in self.config['workflow_parameters']['image_format'].split(',') for p in glob.glob(os.path.join(self.images_dir, '**', f'*.{ext}'), recursive=True)])
        
        # Setup classes and colors
        self.classes_dict = self.config['model_configurations']['classes']
        self.colors = {c: ((c*55+50)%256, (c*95+100)%256, (c*135+150)%256) for c in self.classes_dict.keys()}
        
        # Initialize review file list
        self.review_file_path = os.path.join(self.dataset_dir, "review.txt")
        self.review_files = set()
        self._load_review_files()

        # Initialize image index
        self.img_index = 0
        if self.args.start_image:
            try:
                self.img_index = [os.path.basename(p) for p in self.image_paths].index(self.args.start_image)
            except ValueError:
                print(f"Warning: Start image '{self.args.start_image}' not found. Starting from the beginning.")

    def _load_review_files(self):
        """Load image filenames from review.txt into a set."""
        if os.path.exists(self.review_file_path):
            with open(self.review_file_path, 'r') as f:
                self.review_files = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(self.review_files)} files from {self.review_file_path}")

    def _save_review_files(self):
        """
        Save the current set of review image filenames to review.txt.
        If the review list is empty, this function does nothing.
        """
        # Only create the file if there are items in the review list.
        if not self.review_files:
            print("\nReview list is empty. No review.txt file will be created.")
            return

        with open(self.review_file_path, 'w') as f:
            for fname in sorted(list(self.review_files)):
                f.write(f"{fname}\n")
        print(f"\nSaved {len(self.review_files)} files to {self.review_file_path}.")

    def _mouse_callback(self, event, x, y, flags, param):
        """Handles mouse click events to add/remove images from the review list."""
        if event == cv2.EVENT_LBUTTONDOWN:
            img_name = os.path.basename(self.image_paths[self.img_index])
            action = "added" if img_name not in self.review_files else "removed"
            
            if action == "added":
                self.review_files.add(img_name)
            else:
                self.review_files.remove(img_name)
            
            print(f"  -> '{img_name}' {action} to review list (Total: {len(self.review_files)})")
            # Redraw the current image immediately to reflect the change
            self._draw_and_show_image()

    def _draw_and_show_image(self):
        """Reads, draws labels and review status, and displays the current image."""
        img_path = self.image_paths[self.img_index]
        img_name = os.path.basename(img_path)
        label_path = os.path.splitext(img_path.replace(self.images_dir, self.labels_dir, 1))[0] + '.txt'
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load '{img_name}'. Skipping.")
            return

        h, w = img.shape[:2]
        
        # Draw bounding boxes from label files
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cid, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                    x1, y1, x2, y2 = int((xc-bw/2)*w), int((yc-bh/2)*h), int((xc+bw/2)*w), int((yc+bh/2)*h)
                    color = self.colors.get(cid, (255, 255, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f'{cid}:{self.classes_dict.get(cid, "?")}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(img, "LABEL NOT FOUND", (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)
            
        # Display "REVIEW" text if the image is in the review list
        if img_name in self.review_files:
            cv2.putText(img, "REVIEW", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Label Visualizer", img)
        cv2.setWindowTitle("Label Visualizer", f"[{self.img_index+1}/{len(self.image_paths)}] {img_name}")

    def run(self):
        """Main loop to run the visualizer."""
        if not self.image_paths:
            print(f"Error: No images found in '{self.images_dir}'.")
            return

        print("\n" + "="*50); print("Starting Dataset Visualizer."); print("="*50)
        print(f"  - Target Dataset: {self.dataset_dir}")
        if self.args.start_image: print(f"  - Starting Image: {self.args.start_image}")
        print("="*50)
        print("\nControls:")
        print("  [d] or [Space]: Next Image")
        print("  [a]: Previous Image")
        print("  [Mouse Left Click]: Add/Remove from Review List")
        print("  [q]: Quit and Save Review List")

        # Create a window and set the mouse callback function
        cv2.namedWindow("Label Visualizer", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Label Visualizer", self._mouse_callback)

        while True:
            self._draw_and_show_image()
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('d') or key == ord(' '):
                if self.img_index < len(self.image_paths) - 1:
                    self.img_index += 1
            elif key == ord('a'):
                if self.img_index > 0:
                    self.img_index -= 1
        
        # Cleanup
        self._save_review_files()
        cv2.destroyAllWindows()
        print("Visualizer closed.")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: _config.yaml not found.")
        exit()

    parser = argparse.ArgumentParser(description="Visualizes labels for a dataset and manages a review list.")
    parser.add_argument('--dataset', type=str, default=None, help="Relative path to the dataset to visualize.")
    parser.add_argument('--start_image', type=str, default=None, help="Filename of the image to start visualization from.")
    args = parser.parse_args()

    visualizer = DatasetVisualizer(config, args)
    visualizer.run()