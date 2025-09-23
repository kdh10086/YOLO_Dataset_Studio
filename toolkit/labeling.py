import cv2
import os
import sys
import glob
import shutil
import numpy as np
from collections import Counter
import torch
from ultralytics import YOLO
from tqdm import tqdm

from toolkit.utils import get_label_path

class IntegratedLabeler:
    """
    An advanced GUI for creating and reviewing bounding box labels with features
    like point-to-point drawing, a zoom magnifier, and dynamic UI feedback.
    """
    def __init__(self, dataset_dir, config):
        # --- Core Setup ---
        self.dataset_dir = dataset_dir
        self.config = config
        self.classes = config.get('model_configurations', {}).get('classes', {0: 'object'})
        # Generate distinct colors for each class
        self.colors = {c: ((c*55+50)%256, (c*95+100)%256, (c*135+150)%256) for c in self.classes.keys()}

        # --- Image and Data State ---
        self.image_paths, self.filtered_image_indices = [], []
        self.img_index, self.current_class_id = 0, 0
        self.current_bboxes, self.review_list, self.history = [], set(), []
        self.img_orig, self.display_img, self.clone = None, None, None
        self.h_orig, self.w_orig, self.ratio = 0, 0, 1.0

        # --- UI State ---
        self.quit_flag = False
        self.mode, self.filter_mode = 'draw', 'all'
        self.window_name = "Integrated Labeler"
        self.display_width = 1280 # Target width for the main display

        # --- Point-to-Point Drawing State ---
        self.first_point = None
        self.current_mouse_pos = None

        # --- Magnifier Window State ---
        self.magnifier_window_name = "Magnifier"
        self.magnifier_size = 680
        self.magnifier_zoom_level = 4 # e.g., 4x zoom

    def _calculate_iou(self, box1, box2):
        # box format: [x1, y1, x2, y2]
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        if inter_area == 0:
            return 0.0

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area

    def _draw_dotted_rectangle(self, img, pt1, pt2, color, thickness, gap=10):
        """Draws a dotted rectangle, used for the deletion mode indicator."""
        s, e = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1])), (max(pt1[0], pt2[0]), max(pt1[1], pt2[1]))
        # Draw horizontal lines
        for i in range(s[0], e[0], gap * 2):
            cv2.line(img, (i, s[1]), (min(i + gap, e[0]), s[1]), color, thickness)
            cv2.line(img, (i, e[1]), (min(i + gap, e[0]), e[1]), color, thickness)
        # Draw vertical lines
        for i in range(s[1], e[1], gap * 2):
            cv2.line(img, (s[0], i), (s[0], min(i + gap, e[1])), color, thickness)
            cv2.line(img, (e[0], i), (e[0], min(i + gap, e[1])), color, thickness)

    def _pixels_to_yolo(self, bbox):
        cid,x1,y1,x2,y2=bbox; return cid,((x1+x2)/2)/self.w_orig,((y1+y2)/2)/self.h_orig,abs(x2-x1)/self.w_orig,abs(y2-y1)/self.h_orig

    def _load_data(self):
        fmts = self.config.get('workflow_parameters', {}).get('image_format', 'png,jpg,jpeg').split(',')

        print("[Info] Searching for images in all subdirectories...")
        # Find all image files recursively within the dataset directory.
        all_image_files = sorted([
            p for ext in fmts
            for p in glob.glob(os.path.join(self.dataset_dir, '**', f'*.{ext}'), recursive=True)
        ])

        # Filter these paths to only include those located within a directory named 'images'.
        # This robustly handles both '.../images/train/...' and '.../train/images/...' structures.
        sep = os.path.sep
        self.image_paths = [
            path for path in all_image_files
            if f'{sep}images{sep}' in path
        ]

        if not self.image_paths:
            print(f"[Error] No images found within any 'images' subdirectory in '{self.dataset_dir}'.")
            print("Please ensure your dataset follows a standard structure, such as:")
            print("1. dataset/images/{train,val}/...")
            print("2. dataset/{train,val}/images/...")
            return False

        print(f"[Info] Found {len(self.image_paths)} images.")
        rev_path = os.path.join(self.dataset_dir, 'review_list.txt')
        if os.path.exists(rev_path):
            self.review_list = {ln.strip() for ln in open(rev_path, 'r') if ln.strip()}
        self._apply_filter()
        return True

    def _save_current_labels(self):
        if self.img_index < len(self.image_paths):
            lbl_path = get_label_path(self.image_paths[self.img_index])
            os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
            with open(lbl_path, 'w') as f: f.write('\n'.join([f"{b[0]} {self._pixels_to_yolo(b)[1]:.6f} {self._pixels_to_yolo(b)[2]:.6f} {self._pixels_to_yolo(b)[3]:.6f} {self._pixels_to_yolo(b)[4]:.6f}" for b in self.current_bboxes]))

    def _save_review_list(self):
        p = os.path.join(self.dataset_dir, 'review_list.txt')
        if self.review_list: open(p,'w').write('\n'.join(sorted(list(self.review_list))))
        elif os.path.exists(p): os.remove(p)

    def _isolate_current_image(self):
        p=self.image_paths[self.img_index]; lp=get_label_path(p); iso_dir=os.path.join(self.dataset_dir,'_isolated')
        [os.makedirs(os.path.join(iso_dir,d),exist_ok=True) for d in ['images','labels']]
        shutil.move(p,os.path.join(iso_dir,'images',os.path.basename(p)))
        if os.path.exists(lp): shutil.move(lp,os.path.join(iso_dir,'labels',os.path.basename(lp)))
        self.image_paths.pop(self.img_index); self._apply_filter(); self.img_index=min(self.img_index,len(self.filtered_image_indices)-1)

    def _redraw_ui(self):
        self.clone = self.display_img.copy()
        h, w, _ = self.clone.shape

        # Draw existing bounding boxes
        for cid, x1, y1, x2, y2 in self.current_bboxes:
            color = self.colors.get(cid, (0, 255, 0))
            cv2.rectangle(self.clone, (int(x1 * self.ratio), int(y1 * self.ratio)), (int(x2 * self.ratio), int(y2 * self.ratio)), color, 2)

        # Draw point-to-point preview
        if self.first_point and self.current_mouse_pos:
            start_point_scaled = (int(self.first_point[0] * self.ratio), int(self.first_point[1] * self.ratio))
            end_point_scaled = self.current_mouse_pos
            if self.mode == 'draw':
                color = self.colors.get(self.current_class_id, (0, 255, 0))
                cv2.rectangle(self.clone, start_point_scaled, end_point_scaled, color, 2)
                # Display class name next to the cursor while drawing
                class_name = self.classes.get(self.current_class_id, "Unknown")
                text_pos = (end_point_scaled[0], end_point_scaled[1] - 10)
                cv2.putText(self.clone, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else: # 'delete' mode
                color = (0, 0, 255) # Red
                self._draw_dotted_rectangle(self.clone, start_point_scaled, end_point_scaled, color, 2)

        # Draw crosshairs
        if self.current_mouse_pos:
            mx, my = self.current_mouse_pos
            color = (0, 0, 255) if self.mode == 'delete' else self.colors.get(self.current_class_id, (0, 255, 255))
            cv2.line(self.clone, (mx, 0), (mx, h), color, 1)
            cv2.line(self.clone, (0, my), (w, my), color, 1)


        # Add review flag indicator
        if os.path.basename(self.image_paths[self.img_index]) in self.review_list:
            cv2.putText(self.clone, "R", (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 3)

    def _update_magnifier(self):
        if not self.current_mouse_pos or self.img_orig is None:
            magnifier_img = np.zeros((self.magnifier_size, self.magnifier_size, 3), dtype=np.uint8)
            cv2.putText(magnifier_img, "No Image", (150, 340), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imshow(self.magnifier_window_name, magnifier_img)
            return

        mx_orig, my_orig = int(self.current_mouse_pos[0] / self.ratio), int(self.current_mouse_pos[1] / self.ratio)

        # Define the size of the region to crop before zooming
        crop_w = int(self.magnifier_size / self.magnifier_zoom_level)
        crop_h = int(self.magnifier_size / self.magnifier_zoom_level)

        # Calculate the ideal top-left corner of the crop area (can be negative)
        x1_ideal = mx_orig - crop_w // 2
        y1_ideal = my_orig - crop_h // 2

        # Create a black canvas that will contain the crop
        padded_crop = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        # Determine the overlapping region between the ideal crop and the actual image
        x_src_start = max(0, x1_ideal)
        y_src_start = max(0, y1_ideal)
        x_src_end = min(self.w_orig, x1_ideal + crop_w)
        y_src_end = min(self.h_orig, y1_ideal + crop_h)

        # Determine where to place the valid image region onto the black canvas
        x_dst_start = max(0, -x1_ideal)
        y_dst_start = max(0, -y1_ideal)

        # Get the width and height of the actual region to copy
        copy_w = x_src_end - x_src_start
        copy_h = y_src_end - y_src_start

        # Copy the valid image data to the canvas if there is an overlap
        if copy_w > 0 and copy_h > 0:
            padded_crop[y_dst_start:y_dst_start+copy_h, x_dst_start:x_dst_start+copy_w] = \
                self.img_orig[y_src_start:y_src_end, x_src_start:x_src_end]

        # Resize the padded crop to the final magnifier size. This prevents stretching.
        magnifier_img = cv2.resize(padded_crop, (self.magnifier_size, self.magnifier_size), interpolation=cv2.INTER_NEAREST)
        h_mag, w_mag, _ = magnifier_img.shape

        # This function converts original image coordinates to the new magnifier view's coordinates
        def to_mag_coords(p):
            px, py = p
            # The transformation is relative to the top-left of the ideal (padded) crop box
            return int((px - x1_ideal) * self.magnifier_zoom_level), int((py - y1_ideal) * self.magnifier_zoom_level)

        # Draw bounding boxes that are visible in the magnified region
        for cid, b_x1, b_y1, b_x2, b_y2 in self.current_bboxes:
            if b_x2 > x1_ideal and b_x1 < (x1_ideal + crop_w) and b_y2 > y1_ideal and b_y1 < (y1_ideal + crop_h):
                color = self.colors.get(cid, (0, 255, 0))
                cv2.rectangle(magnifier_img, to_mag_coords((b_x1, b_y1)), to_mag_coords((b_x2, b_y2)), color, 2)

        # Draw the preview rectangle if a box is being drawn
        if self.first_point:
            if self.mode == 'draw':
                color = self.colors.get(self.current_class_id, (0, 255, 0))
            else: # delete mode
                color = (0, 0, 255) # Red
            cv2.rectangle(magnifier_img, to_mag_coords(self.first_point), to_mag_coords((mx_orig, my_orig)), color, 2)

        # Draw a dynamic crosshair in the center of the magnifier
        crosshair_color = (0, 0, 255) if self.mode == 'delete' else self.colors.get(self.current_class_id, (0, 255, 255))
        cv2.line(magnifier_img, (w_mag // 2, 0), (w_mag // 2, h_mag), crosshair_color, 2)
        cv2.line(magnifier_img, (0, h_mag // 2), (w_mag, h_mag // 2), crosshair_color, 2)

        cv2.imshow(self.magnifier_window_name, magnifier_img)

    def _handle_mouse(self, event, x, y, flags, param):
        self.current_mouse_pos = (x, y)
        ox, oy = int(x / self.ratio), int(y / self.ratio)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.first_point is None:
                self.first_point = (ox, oy)
            else:
                self.history.append(self.current_bboxes.copy())
                # Finalize the rectangle
                rect_x1, rect_y1 = min(self.first_point[0], ox), min(self.first_point[1], oy)
                rect_x2, rect_y2 = max(self.first_point[0], ox), max(self.first_point[1], oy)

                if self.mode == 'draw':
                    self.current_bboxes.append((self.current_class_id, rect_x1, rect_y1, rect_x2, rect_y2))
                elif self.mode == 'delete':
                    initial_box_count = len(self.current_bboxes)
                    # Keep boxes whose center is NOT within the deletion rectangle
                    self.current_bboxes = [
                        b for b in self.current_bboxes
                        if not (rect_x1 < (b[1] + b[3]) / 2 < rect_x2 and
                                rect_y1 < (b[2] + b[4]) / 2 < rect_y2)
                    ]
                    removed_count = initial_box_count - len(self.current_bboxes)
                    if removed_count > 0:
                        print(f"-> Removed {removed_count} boxes.")

                self.first_point = None # Reset for next operation

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.first_point is not None:
                self.first_point = None

    def _apply_filter(self):
        if self.filter_mode=='review': self.filtered_image_indices=[i for i,p in enumerate(self.image_paths) if os.path.basename(p) in self.review_list]
        else: self.filtered_image_indices=list(range(len(self.image_paths)))
        if not self.filtered_image_indices: self.filtered_image_indices=list(range(len(self.image_paths)))

    def _navigate(self,d):
        self._save_current_labels()
        try: cur=self.filtered_image_indices.index(self.img_index)
        except ValueError: cur=0
        self.img_index=self.filtered_image_indices[max(0,min(cur+d,len(self.filtered_image_indices)-1))]

    def _load_image_and_labels(self):
        if not (0 <= self.img_index < len(self.image_paths)):
            return False
        p = self.image_paths[self.img_index]
        self.img_orig = cv2.imread(p)
        if self.img_orig is None: return False

        self.current_bboxes, self.history, self.first_point = [], [], None

        self.h_orig, self.w_orig = self.img_orig.shape[:2]

        self.ratio = self.display_width / self.w_orig
        self.display_img = cv2.resize(self.img_orig, (self.display_width, int(self.h_orig * self.ratio)))

        lp = get_label_path(p)
        if os.path.exists(lp):
            with open(lp, 'r') as f:
                lines = [ln.strip().split() for ln in f.readlines()]
                self.current_bboxes = [
                    (int(l[0]), *map(int, [
                        (float(l[1]) - float(l[3]) / 2) * self.w_orig,
                        (float(l[2]) - float(l[4]) / 2) * self.h_orig,
                        (float(l[1]) + float(l[3]) / 2) * self.w_orig,
                        (float(l[2]) + float(l[4]) / 2) * self.h_orig
                    ])) for l in lines if len(l) == 5
                ]
        return True

    def run(self):
        if not self._load_data(): return

        print("\n" + "="*50)
        print(" " * 12 + "Integrated Labeler Controls")
        print("="*50)
        print(" Modes & Drawing:")
        print("  - [W]: Draw Mode (Default)")
        print("  - [E]: Delete Mode")
        print("  - [R]: Undo Last Action")
        print("  - [1-9]: Select Class")
        print("-" * 50)
        print(" Navigation & Saving:")
        print("  - [D]: Save & Next Image")
        print("  - [A]: Save & Previous Image")
        print("  - [Q]: Save & Quit")
        print("  - [C]: Force Quit (discards current file's changes)")
        print("-" * 50)
        print(" Workflow:")
        print("  - [F]: Flag / Unflag for Review")
        print("  - [T]: Toggle Filter (All / Review)")
        print("  - [X]: Exclude Current Image")
        print("="*50)

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._handle_mouse)
        cv2.namedWindow(self.magnifier_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.magnifier_window_name, self.magnifier_size, self.magnifier_size)

        while not self.quit_flag:
            if not self._load_image_and_labels():
                if self.image_paths: self._navigate(1); continue
                else: print("[Info] No more images to label."); break

            while True:
                img_name = os.path.basename(self.image_paths[self.img_index])
                try:
                    progress = f"({self.filtered_image_indices.index(self.img_index) + 1}/{len(self.filtered_image_indices)})"
                except ValueError:
                    progress = "(0/0)"

                title = f"Labeler {progress} - {img_name} - MODE: {self.mode.upper()}"
                if self.mode == 'draw':
                    title += f" (Class: {self.classes.get(self.current_class_id, 'N/A')})"
                cv2.setWindowTitle(self.window_name, title)

                self._redraw_ui()
                self._update_magnifier()

                cv2.imshow(self.window_name, self.clone)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), ord('c')]:
                    if key != ord('c'): self._save_current_labels()
                    self.quit_flag = True; break
                elif key in [ord('d'), ord('s')]: # Next
                    self._save_current_labels(); self._navigate(1); break
                elif key == ord('a'): # Previous
                    self._save_current_labels(); self._navigate(-1); break
                elif key == ord('r'): # Undo
                    if self.history: self.current_bboxes = self.history.pop()
                elif ord('1') <= key <= ord('9'):
                    class_id = int(chr(key)) - 1
                    if class_id in self.classes:
                        self.current_class_id = class_id
                        print(f"Current class set to: {self.classes.get(self.current_class_id, 'N/A')}")
                        if self.mode == 'delete':
                            self.mode = 'draw'
                            print("-> Switched to Draw Mode")
                    else:
                        print(f"[Warning] Class ID {class_id} is not defined in config.")
                elif key == ord('w'): self.mode = 'draw'
                elif key == ord('e'): self.mode = 'delete'
                elif key == ord('f'): # Flag for review
                    n = os.path.basename(self.image_paths[self.img_index])
                    if n in self.review_list: self.review_list.remove(n)
                    else: self.review_list.add(n)
                elif key == ord('x'): # Exclude
                    self._isolate_current_image(); break
                elif key == ord('t'): # Toggle filter
                    self.filter_mode = 'review' if self.filter_mode == 'all' else 'all'
                    self._apply_filter()
                    if self.filtered_image_indices: self.img_index = self.filtered_image_indices[0]
                    break

        self._save_review_list()
        cv2.destroyAllWindows()

def launch_labeler(dataset_dir,config): IntegratedLabeler(dataset_dir,config).run()

def auto_label_dataset(dataset_path, weights_path, config):
    # Safely get configuration with fallbacks
    workflow_params = config.get('workflow_parameters', {})
    model_configs = config.get('model_configurations', {})
    teacher_config = model_configs.get('teacher_model_config', {})
    hyperparams = teacher_config.get('hyperparameters', {})
    models_params = hyperparams.get('models', {})

    conf = workflow_params.get('auto_label_confidence_threshold', 0.3)
    model_name = teacher_config.get('model_name', 'default')

    # Robustly get batch size, falling back to default
    model_specific_params = models_params.get(model_name, models_params.get('default', {}))
    batch = model_specific_params.get('batch_size', 16)

    if not all([conf, model_name, batch]):
        print("[Error] Could not find all required parameters in config for auto-labeling.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Auto-labeling on device: {device}")
    model = YOLO(weights_path).to(device)

    # --- Flexible Image Path Discovery ---
    image_formats = workflow_params.get('image_format', 'png,jpg,jpeg').split(',')
    print("[Info] Searching for images in all subdirectories for auto-labeling...")
    
    all_image_files = sorted([
        p for fmt in image_formats
        for p in glob.glob(os.path.join(dataset_path, '**', f'*.{fmt}'), recursive=True)
    ])

    sep = os.path.sep
    paths = [
        path for path in all_image_files
        if f'{sep}images{sep}' in path
    ]
    
    if not paths:
        print("[Warning] No images found to label within any 'images' subdirectory.")
        return

    print(f"Found {len(paths)} images to process...")
    for i in tqdm(range(0, len(paths), batch), desc="Auto-labeling"):
        batch_paths = paths[i:i+batch]
        try:
            results = model(batch_paths, conf=conf, verbose=False)

            for r in results:
                if not r.boxes:
                    continue

                # --- Robust Label Path Generation ---
                output_path = get_label_path(r.path)
                # Ensure the directory for the label file exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                lines = []
                for box in r.boxes:
                    if box.xywhn.nelement() > 0:
                        xywhn = box.xywhn[0]
                        line = f"{int(box.cls)} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}"
                        lines.append(line)

                if lines:
                    with open(output_path, 'w') as f:
                        f.write('\n'.join(lines))

        except Exception as e:
            print(f"\n[Error] Failed to process batch starting with {os.path.basename(batch_paths[0])}: {e}")

    print("\nAuto-labeling complete.")