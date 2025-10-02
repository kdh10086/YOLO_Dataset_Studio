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

from toolkit.utils import build_class_hotkeys, get_label_path, normalize_class_map

class IntegratedLabeler:
    """
    An advanced GUI for creating and reviewing bounding box labels with features
    like point-to-point drawing, a zoom magnifier, and dynamic UI feedback.
    """
    def __init__(self, dataset_dir, config):
        # --- Core Setup ---
        self.dataset_dir = dataset_dir
        self.config = config

        raw_classes = config.get('model_configurations', {}).get('classes', {0: 'object'})
        normalized_classes = normalize_class_map(raw_classes)
        if not normalized_classes:
            if raw_classes:
                try:
                    from collections.abc import Mapping, Iterable
                    if isinstance(raw_classes, Mapping):
                        fallback_iter = raw_classes.values()
                    elif isinstance(raw_classes, Iterable) and not isinstance(raw_classes, (str, bytes)):
                        fallback_iter = raw_classes
                    else:
                        fallback_iter = []
                except Exception:
                    fallback_iter = []

                normalized_classes = {idx: str(name) for idx, name in enumerate(fallback_iter)}
            if not normalized_classes:
                normalized_classes = {0: 'object'}
        self.classes = normalized_classes

        self.class_hotkeys = build_class_hotkeys(self.classes)
        self.hotkey_to_class = {key: class_id for key, class_id, _ in self.class_hotkeys}

        default_class = next(iter(self.classes)) if self.classes else 0
        if self.hotkey_to_class:
            default_class = self.hotkey_to_class.get(1, default_class)

        # Generate distinct colors for each class
        self.colors = {c: ((c*55+50)%256, (c*95+100)%256, (c*135+150)%256) for c in self.classes.keys()}

        # Overlap detection threshold (clamped into a valid IoU range)
        workflow_cfg = config.get('workflow_parameters', {}) if isinstance(config, dict) else {}
        threshold = workflow_cfg.get('labeling_overlap_iou_threshold', 0.80)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = 0.80
        self.overlap_iou_threshold = max(0.0, min(1.0, threshold))

        # --- Image and Data State ---
        self.primary_image_paths, self.isolated_image_paths = [], []
        self.image_paths, self.filtered_image_indices = [], []
        self.img_index = 0
        self.current_class_id = default_class
        self.current_bboxes, self.review_list, self.history = [], set(), []
        self.bbox_precise = []
        self.img_orig, self.display_img, self.clone = None, None, None
        self.h_orig, self.w_orig, self.ratio = 0, 0, 1.0

        # --- Display Preferences ---
        self.show_class_names = False

        # --- UI State ---
        self.quit_flag = False
        self.mode, self.filter_mode = 'draw', 'all'
        self.viewing_isolated = False
        self.last_primary_index = 0
        self.last_isolated_index = 0
        self.window_name = "Integrated Labeler"
        self.display_width = 1280 # Target width for the main display

        # --- Point-to-Point Drawing State ---
        self.first_point = None
        self.current_mouse_pos = None

        # --- Magnifier Window State ---
        self.magnifier_window_name = "Magnifier"
        self.magnifier_size = 680
        self.magnifier_zoom_level = 4 # e.g., 4x zoom
        
        # --- Clipboard-style buffer & selection state ---
        self.clipboard_bbox = None
        self.active_bbox_index = None
        self.bbox_nudge_step = 1
        self.arrow_key_codes = {
            'left': {2424832, 65361, 63234},
            'right': {2555904, 65363, 63235},
            'up': {2490368, 65362, 63232},
            'down': {2621440, 65364, 63233}
        }
        self._overlap_markers = {}

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
        cid, x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) / 2) / self.w_orig if self.w_orig else 0
        cy = ((y1 + y2) / 2) / self.h_orig if self.h_orig else 0
        bw = abs(x2 - x1) / self.w_orig if self.w_orig else 0
        bh = abs(y2 - y1) / self.h_orig if self.h_orig else 0
        return cid, cx, cy, bw, bh

    def _yolo_to_pixels(self, cls_id, cx, cy, bw, bh):
        if self.w_orig <= 0 or self.h_orig <= 0:
            return None

        px = cx * self.w_orig
        py = cy * self.h_orig
        half_w = (bw * self.w_orig) / 2
        half_h = (bh * self.h_orig) / 2

        x1 = int(round(px - half_w))
        y1 = int(round(py - half_h))
        x2 = int(round(px + half_w))
        y2 = int(round(py + half_h))

        return self._sanitize_bbox_for_current_image((cls_id, x1, y1, x2, y2))

    def _load_data(self):
        fmts = self.config.get('workflow_parameters', {}).get('image_format', 'png,jpg,jpeg').split(',')

        print("[Info] Searching for images in all subdirectories...")
        # Find all image files recursively within the dataset directory.
        all_image_files = sorted([
            p for ext in fmts
            for p in glob.glob(os.path.join(self.dataset_dir, '**', f'*.{ext}'), recursive=True)
        ])

        # Split into the active workspace and the isolated workspace.
        sep = os.path.sep
        self.primary_image_paths = [
            path for path in all_image_files
            if f'{sep}images{sep}' in path and f'{sep}_isolated{sep}' not in path
        ]
        self.isolated_image_paths = [
            path for path in all_image_files
            if f'{sep}_isolated{sep}' in path and f'{sep}images{sep}' in path
        ]

        self.viewing_isolated = False
        self.image_paths = self.primary_image_paths

        if not self.image_paths:
            if self.isolated_image_paths:
                print("[Info] No active dataset images found. Switching to isolated workspace view.")
                self.viewing_isolated = True
                self._apply_filter()
                return True
            print(f"[Error] No images found within any 'images' subdirectory in '{self.dataset_dir}'.")
            print("Please ensure your dataset follows a standard structure, such as:")
            print("1. dataset/images/{train,val}/...")
            print("2. dataset/{train,val}/images/...")
            return False

        print(f"[Info] Found {len(self.primary_image_paths)} images.")
        rev_path = os.path.join(self.dataset_dir, 'review_list.txt')
        if os.path.exists(rev_path):
            self.review_list = {ln.strip() for ln in open(rev_path, 'r') if ln.strip()}
        self._apply_filter()
        return True

    def _save_current_labels(self):
        if self.img_index < len(self.image_paths):
            lbl_path = get_label_path(self.image_paths[self.img_index])
            os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
            lines = []
            for bbox in self.current_bboxes:
                yolo_fmt = self._pixels_to_yolo(bbox)
                lines.append(f"{yolo_fmt[0]} {yolo_fmt[1]:.6f} {yolo_fmt[2]:.6f} {yolo_fmt[3]:.6f} {yolo_fmt[4]:.6f}")
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(lines))

    def _save_review_list(self):
        p = os.path.join(self.dataset_dir, 'review_list.txt')
        if self.review_list: open(p,'w').write('\n'.join(sorted(list(self.review_list))))
        elif os.path.exists(p): os.remove(p)

    def _sanitize_bbox_for_current_image(self, bbox):
        if not bbox or self.w_orig <= 0 or self.h_orig <= 0:
            return None

        cid, x1, y1, x2, y2 = bbox
        x1, x2 = sorted((int(round(x1)), int(round(x2))))
        y1, y2 = sorted((int(round(y1)), int(round(y2))))

        max_x = max(self.w_orig - 1, 0)
        max_y = max(self.h_orig - 1, 0)

        x1 = max(0, min(max_x, x1))
        x2 = max(0, min(max_x, x2))
        y1 = max(0, min(max_y, y1))
        y2 = max(0, min(max_y, y2))

        if x1 == x2 or y1 == y2:
            return None

        return (cid, x1, y1, x2, y2)

    def _push_history(self):
        self.history.append((self.current_bboxes.copy(), self.bbox_precise.copy()))
        return len(self.history) - 1

    def _ensure_precise_state(self):
        if len(self.bbox_precise) != len(self.current_bboxes):
            self.bbox_precise = [self._pixels_to_yolo(b) for b in self.current_bboxes]

    def _update_overlap_counts(self):
        """Track overlapping clusters and mark the earliest drawn box only."""
        markers = {}
        threshold = self.overlap_iou_threshold
        num_boxes = len(self.current_bboxes)

        if num_boxes <= 1 or threshold > 1.0:
            self._overlap_markers = {}
            return

        neighbors = [set() for _ in range(num_boxes)]
        coords = [(bbox[1], bbox[2], bbox[3], bbox[4]) for bbox in self.current_bboxes]

        for i in range(num_boxes):
            xi1, yi1, xi2, yi2 = coords[i]
            for j in range(i + 1, num_boxes):
                xj1, yj1, xj2, yj2 = coords[j]
                iou = self._calculate_iou((xi1, yi1, xi2, yi2), (xj1, yj1, xj2, yj2))
                if iou >= threshold:
                    neighbors[i].add(j)
                    neighbors[j].add(i)

        visited = set()
        for i in range(num_boxes):
            if i in visited:
                continue

            if not neighbors[i]:
                visited.add(i)
                continue

            stack = [i]
            component = []
            while stack:
                idx = stack.pop()
                if idx in visited:
                    continue
                visited.add(idx)
                component.append(idx)
                stack.extend(neighbors[idx])

            if len(component) <= 1:
                continue

            anchor_idx = min(component)
            _, ax1, ay1, ax2, ay2 = self.current_bboxes[anchor_idx]
            key = (int(ax1), int(ay1), int(ax2), int(ay2))
            markers[key] = len(component)

        self._overlap_markers = markers

    def _move_last_bbox(self, dx, dy):
        if not self.current_bboxes:
            self.active_bbox_index = None
            return

        self._ensure_precise_state()

        if self.active_bbox_index is None or not (0 <= self.active_bbox_index < len(self.current_bboxes)):
            self.active_bbox_index = len(self.current_bboxes) - 1

        cid, x1, y1, x2, y2 = self.current_bboxes[self.active_bbox_index]

        dx = max(-x1, min(dx, self.w_orig - 1 - x2))
        dy = max(-y1, min(dy, self.h_orig - 1 - y2))

        if dx == 0 and dy == 0:
            return

        history_idx = self._push_history()
        moved_bbox = (cid, x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        self.current_bboxes[self.active_bbox_index] = moved_bbox
        self.clipboard_bbox = moved_bbox
        if len(self.bbox_precise) > self.active_bbox_index:
            self.bbox_precise[self.active_bbox_index] = self._pixels_to_yolo(moved_bbox)
        self._update_overlap_counts()

        return history_idx

    def _scale_last_bbox(self, grow):
        if not self.current_bboxes:
            self.active_bbox_index = None
            return

        self._ensure_precise_state()

        if self.active_bbox_index is None or not (0 <= self.active_bbox_index < len(self.current_bboxes)):
            self.active_bbox_index = len(self.current_bboxes) - 1

        cid, x1, y1, x2, y2 = self.current_bboxes[self.active_bbox_index]
        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1 or self.w_orig <= 0 or self.h_orig <= 0:
            return

        short_side = min(w, h)
        long_side = max(w, h)

        if grow:
            new_short = short_side + 1
        else:
            if short_side <= 1:
                return
            new_short = short_side - 1

        scale = new_short / short_side if short_side > 0 else None
        if scale is None or scale <= 0:
            return

        new_long = long_side * scale

        if w <= h:
            new_w = new_short
            new_h = max(1, int(round(new_long)))
        else:
            new_h = new_short
            new_w = max(1, int(round(new_long)))

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        new_x1 = int(round(cx - new_w / 2.0))
        new_x2 = new_x1 + new_w
        new_y1 = int(round(cy - new_h / 2.0))
        new_y2 = new_y1 + new_h

        scaled = self._sanitize_bbox_for_current_image((cid, new_x1, new_y1, new_x2, new_y2))
        if scaled is None:
            return

        scaled_w = scaled[3] - scaled[1]
        scaled_h = scaled[4] - scaled[2]
        if scaled_w <= 0 or scaled_h <= 0:
            return

        history_idx = self._push_history()
        self.current_bboxes[self.active_bbox_index] = scaled
        self.clipboard_bbox = scaled
        if len(self.bbox_precise) > self.active_bbox_index:
            self.bbox_precise[self.active_bbox_index] = self._pixels_to_yolo(scaled)
        self._update_overlap_counts()

        return history_idx

    def _scale_last_bbox_width(self, grow):
        """Adjust only the width of the most recent bounding box."""
        if not self.current_bboxes:
            self.active_bbox_index = None
            return

        self._ensure_precise_state()

        if self.active_bbox_index is None or not (0 <= self.active_bbox_index < len(self.current_bboxes)):
            self.active_bbox_index = len(self.current_bboxes) - 1

        cid, x1, y1, x2, y2 = self.current_bboxes[self.active_bbox_index]
        width = x2 - x1
        if width <= 1:
            return

        if not grow and width <= 2:
            return

        delta = 1 if grow else -1
        new_x1 = x1 - delta
        new_x2 = x2 + delta

        scaled = self._sanitize_bbox_for_current_image((cid, new_x1, y1, new_x2, y2))
        if scaled is None:
            return

        scaled_width = scaled[3] - scaled[1]
        if scaled_width <= 1:
            return

        if not grow and scaled_width >= width:
            return
        if grow and scaled_width <= width:
            return

        history_idx = self._push_history()
        self.current_bboxes[self.active_bbox_index] = scaled
        self.clipboard_bbox = scaled
        if len(self.bbox_precise) > self.active_bbox_index:
            self.bbox_precise[self.active_bbox_index] = self._pixels_to_yolo(scaled)
        self._update_overlap_counts()

        return history_idx

    def _isolate_current_image(self):
        if self.viewing_isolated:
            # Return image back to the main dataset using its relative path inside _isolated
            if not self.image_paths:
                return

            current_idx = self.img_index
            image_path = self.image_paths[current_idx]
            rel_inside_iso = os.path.relpath(image_path, os.path.join(self.dataset_dir, '_isolated'))
            target_path = os.path.join(self.dataset_dir, rel_inside_iso)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.move(image_path, target_path)

            label_path = get_label_path(image_path)
            if os.path.exists(label_path):
                rel_label_inside_iso = os.path.relpath(label_path, os.path.join(self.dataset_dir, '_isolated'))
                target_label_path = os.path.join(self.dataset_dir, rel_label_inside_iso)
                os.makedirs(os.path.dirname(target_label_path), exist_ok=True)
                shutil.move(label_path, target_label_path)

            self.image_paths.pop(current_idx)
            self.isolated_image_paths = self.image_paths
            self.primary_image_paths.append(target_path)
            self.primary_image_paths.sort()
            try:
                new_primary_index = self.primary_image_paths.index(target_path)
            except ValueError:
                new_primary_index = len(self.primary_image_paths) - 1
            self.last_primary_index = new_primary_index

            if not self.image_paths:
                self.filtered_image_indices = []
                self.img_index = 0
                self.viewing_isolated = False
                self.image_paths = self.primary_image_paths
                self._apply_filter(announce=True)
                if self.filtered_image_indices:
                    self.img_index = min(self.last_primary_index, len(self.filtered_image_indices) - 1)
                return

            self._apply_filter(announce=True)
            if self.filtered_image_indices:
                if current_idx < len(self.filtered_image_indices):
                    self.img_index = self.filtered_image_indices[current_idx % len(self.filtered_image_indices)]
                else:
                    self.img_index = self.filtered_image_indices[-1]
            else:
                self.img_index = 0
            self._update_overlap_counts()
            self.last_isolated_index = self.img_index
            return

        if not self.image_paths:
            return

        current_idx = self.img_index
        image_path = self.image_paths[current_idx]
        label_path = get_label_path(image_path)
        iso_dir = os.path.join(self.dataset_dir, '_isolated')

        rel_image_path = os.path.relpath(image_path, self.dataset_dir)
        target_image_path = os.path.join(iso_dir, rel_image_path)
        os.makedirs(os.path.dirname(target_image_path), exist_ok=True)
        shutil.move(image_path, target_image_path)

        if os.path.exists(label_path):
            rel_label_path = os.path.relpath(label_path, self.dataset_dir)
            target_label_path = os.path.join(iso_dir, rel_label_path)
            os.makedirs(os.path.dirname(target_label_path), exist_ok=True)
            shutil.move(label_path, target_label_path)

        self.review_list.discard(os.path.basename(image_path))

        self.image_paths.pop(current_idx)
        self.primary_image_paths = self.image_paths  # keep reference explicit
        self.isolated_image_paths.append(target_image_path)
        self.isolated_image_paths.sort()
        try:
            new_iso_index = self.isolated_image_paths.index(target_image_path)
        except ValueError:
            new_iso_index = len(self.isolated_image_paths) - 1
        self.last_isolated_index = new_iso_index

        if not self.image_paths:
            self.filtered_image_indices = []
            self.img_index = 0
            self.current_bboxes = []
            self.bbox_precise = []
            self._overlap_markers = {}
            return

        announce = self.filter_mode != 'all'
        self._apply_filter(announce=announce)

        if self.filtered_image_indices:
            next_candidates = [idx for idx in self.filtered_image_indices if idx >= current_idx]
            if next_candidates:
                self.img_index = next_candidates[0]
            else:
                self.img_index = self.filtered_image_indices[-1]
        else:
            self.img_index = min(current_idx, len(self.image_paths) - 1)
        self._update_overlap_counts()
        self.last_primary_index = self.img_index
        self.last_isolated_index = min(self.last_isolated_index, len(self.isolated_image_paths) - 1) if self.isolated_image_paths else 0

    def _redraw_ui(self):
        self._update_overlap_counts()
        self.clone = self.display_img.copy()
        h, w, _ = self.clone.shape

        # Draw existing bounding boxes
        for cid, x1, y1, x2, y2 in self.current_bboxes:
            color = self.colors.get(cid, (0, 255, 0))
            cv2.rectangle(self.clone, (int(x1 * self.ratio), int(y1 * self.ratio)), (int(x2 * self.ratio), int(y2 * self.ratio)), color, 2)
            if self.show_class_names:
                label = self.classes.get(cid, f"{cid}")
                if label:
                    text_scale = 0.6
                    text_thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
                    corner_x = int(x2 * self.ratio)
                    corner_y = int(y1 * self.ratio)

                    draw_x = corner_x - text_w
                    draw_y = corner_y - 4

                    draw_x = max(0, min(draw_x, w - text_w - 2))
                    if draw_y - text_h < 0:
                        draw_y = text_h + 2
                    if draw_y + baseline >= corner_y:
                        draw_y = max(text_h + 2, corner_y - baseline - 2)

                    cv2.putText(
                        self.clone,
                        label,
                        (draw_x, draw_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        color,
                        text_thickness,
                        cv2.LINE_AA
                    )

        for (x1, y1, x2, y2), count in self._overlap_markers.items():
            text_pos_x = int(round(x2 * self.ratio)) - 10
            text_pos_y = int(round(y1 * self.ratio)) - 5
            text_pos_x = max(0, min(text_pos_x, self.clone.shape[1] - 10))
            text_pos_y = max(15, min(text_pos_y, self.clone.shape[0] - 10))
            cv2.putText(self.clone, str(count), (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
        self._update_overlap_counts()
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

        # Draw overlap counts inside the magnifier view
        for (x1, y1, x2, y2), count in self._overlap_markers.items():
            if x2 > x1_ideal and x1 < (x1_ideal + crop_w) and y2 > y1_ideal and y1 < (y1_ideal + crop_h):
                tr_x, tr_y = to_mag_coords((x2, y1))
                text_pos_x = max(0, min(tr_x - 10, w_mag - 10))
                text_pos_y = max(15, min(tr_y - 5, h_mag - 10))
                cv2.putText(magnifier_img, str(count), (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
                self._ensure_precise_state()
                self._push_history()
                # Finalize the rectangle
                rect_x1, rect_y1 = min(self.first_point[0], ox), min(self.first_point[1], oy)
                rect_x2, rect_y2 = max(self.first_point[0], ox), max(self.first_point[1], oy)

                if self.mode == 'draw':
                    new_bbox = (self.current_class_id, rect_x1, rect_y1, rect_x2, rect_y2)
                    self.current_bboxes.append(new_bbox)
                    self.bbox_precise.append(self._pixels_to_yolo(new_bbox))
                    self.clipboard_bbox = new_bbox
                    self.active_bbox_index = len(self.current_bboxes) - 1
                    self._update_overlap_counts()
                elif self.mode == 'delete':
                    initial_box_count = len(self.current_bboxes)
                    filtered_boxes = []
                    filtered_precise = []
                    for bbox, precise in zip(self.current_bboxes, self.bbox_precise):
                        center_x = (bbox[1] + bbox[3]) / 2
                        center_y = (bbox[2] + bbox[4]) / 2
                        if rect_x1 < center_x < rect_x2 and rect_y1 < center_y < rect_y2:
                            continue
                        filtered_boxes.append(bbox)
                        filtered_precise.append(precise)

                    removed_count = initial_box_count - len(filtered_boxes)
                    self.current_bboxes = filtered_boxes
                    self.bbox_precise = filtered_precise
                    if removed_count > 0:
                        print(f"-> Removed {removed_count} boxes.")
                    self.active_bbox_index = len(self.current_bboxes) - 1 if self.current_bboxes else None
                    self.clipboard_bbox = self.current_bboxes[self.active_bbox_index] if self.current_bboxes else None
                    self._update_overlap_counts()

                self.first_point = None # Reset for next operation

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.first_point is not None:
                self.first_point = None

    def _is_image_unlabeled(self, image_path):
        label_path = get_label_path(image_path)
        if not os.path.exists(label_path):
            return True
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        return False
            return True
        except OSError:
            return True

    def _has_overlapping_boxes(self, image_path):
        label_path = get_label_path(image_path)
        if not os.path.exists(label_path):
            return False

        boxes = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) < 5:
                        continue
                    try:
                        _, cx, cy, bw, bh = map(float, tokens[:5])
                    except ValueError:
                        continue
                    if bw <= 0 or bh <= 0:
                        continue
                    x1 = max(0.0, cx - bw / 2)
                    y1 = max(0.0, cy - bh / 2)
                    x2 = min(1.0, cx + bw / 2)
                    y2 = min(1.0, cy + bh / 2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append((x1, y1, x2, y2))
        except OSError:
            return False

        threshold = self.overlap_iou_threshold
        for i in range(len(boxes)):
            x1_a, y1_a, x2_a, y2_a = boxes[i]
            for j in range(i + 1, len(boxes)):
                x1_b, y1_b, x2_b, y2_b = boxes[j]
                inter_x1 = max(x1_a, x1_b)
                inter_y1 = max(y1_a, y1_b)
                inter_x2 = min(x2_a, x2_b)
                inter_y2 = min(y2_a, y2_b)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    continue
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area_a = (x2_a - x1_a) * (y2_a - y1_a)
                area_b = (x2_b - x1_b) * (y2_b - y1_b)
                union = area_a + area_b - inter_area
                if union <= 0:
                    continue
                iou = inter_area / union
                if iou >= threshold:
                    return True
        return False

    def _apply_filter(self, announce=False, preserve_on_empty=False):
        paths = self.isolated_image_paths if self.viewing_isolated else self.primary_image_paths
        self.image_paths = paths

        if not paths:
            self.filtered_image_indices = []
            if announce:
                view_label = 'Isolated view' if self.viewing_isolated else 'Dataset view'
                print(f"-> {view_label}: No images available.")
            return 0

        if self.filter_mode == 'review' and not self.viewing_isolated:
            indices = [i for i, p in enumerate(paths) if os.path.basename(p) in self.review_list]
        elif self.filter_mode == 'overlap':
            indices = [i for i, p in enumerate(paths) if self._has_overlapping_boxes(p)]
        elif self.filter_mode == 'unlabeled':
            indices = [i for i, p in enumerate(paths) if self._is_image_unlabeled(p)]
        else:
            indices = list(range(len(paths)))

        if not indices:
            if self.filter_mode != 'all':
                if announce:
                    if preserve_on_empty:
                        print(f"[Info] No images matched the '{self.filter_mode}' filter.")
                    else:
                        print(f"[Info] No images matched the '{self.filter_mode}' filter. Reverting to 'all'.")
                if preserve_on_empty:
                    return 0
                self.filter_mode = 'all'
                indices = list(range(len(paths)))
            else:
                indices = list(range(len(paths)))

        self.filtered_image_indices = indices

        if announce:
            mode_label = {
                'all': 'All images',
                'review': 'Review flagged images',
                'overlap': 'Images with overlapping boxes',
                'unlabeled': 'Unlabeled images only'
            }.get(self.filter_mode, self.filter_mode)
            view_label = 'Isolated view' if self.viewing_isolated else 'Dataset view'
            print(f"-> {view_label}: {mode_label} ({len(indices)} images)")

        return len(indices)

    def _navigate(self,d):
        self._save_current_labels()
        try: cur=self.filtered_image_indices.index(self.img_index)
        except ValueError: cur=0
        self.img_index=self.filtered_image_indices[max(0,min(cur+d,len(self.filtered_image_indices)-1))]
        if self.viewing_isolated:
            self.last_isolated_index = self.img_index
        else:
            self.last_primary_index = self.img_index

    def _toggle_isolated_view(self):
        if self.viewing_isolated:
            self.last_isolated_index = self.img_index
            self.viewing_isolated = False
            if self.primary_image_paths:
                self.last_primary_index = min(self.last_primary_index, len(self.primary_image_paths) - 1)
            self.filter_mode = 'all' if self.filter_mode == 'review' else self.filter_mode
            self._apply_filter(announce=True)
            if self.filtered_image_indices:
                target = self.last_primary_index if self.primary_image_paths else 0
                if target in self.filtered_image_indices:
                    self.img_index = target
                else:
                    self.img_index = self.filtered_image_indices[0]
            else:
                self.img_index = 0
            self.last_primary_index = self.img_index
        else:
            if not self.isolated_image_paths:
                print("[Info] No isolated images to display.")
                return
            self.last_primary_index = self.img_index
            self.viewing_isolated = True
            self.filter_mode = 'all'
            self._apply_filter(announce=True)
            self.last_isolated_index = min(self.last_isolated_index, len(self.isolated_image_paths) - 1)
            if self.filtered_image_indices:
                target = self.last_isolated_index
                if target in self.filtered_image_indices:
                    self.img_index = target
                else:
                    self.img_index = self.filtered_image_indices[0]
            else:
                self.img_index = 0
            self.last_isolated_index = self.img_index

    def _load_image_and_labels(self):
        if not (0 <= self.img_index < len(self.image_paths)):
            return False
        p = self.image_paths[self.img_index]
        self.img_orig = cv2.imread(p)
        if self.img_orig is None: return False

        self.current_bboxes, self.history, self.first_point = [], [], None
        self.bbox_precise = []
        self._overlap_markers = {}

        self.h_orig, self.w_orig = self.img_orig.shape[:2]

        self.ratio = self.display_width / self.w_orig
        self.display_img = cv2.resize(self.img_orig, (self.display_width, int(self.h_orig * self.ratio)))

        lp = get_label_path(p)
        if os.path.exists(lp):
            with open(lp, 'r') as f:
                lines = [ln.strip().split() for ln in f.readlines() if ln.strip()]

            parsed_boxes = []
            for l in lines:
                if len(l) != 5:
                    continue
                cls_id = int(float(l[0]))
                try:
                    cx, cy, bw, bh = map(float, l[1:])
                except ValueError:
                    continue
                bbox = self._yolo_to_pixels(cls_id, cx, cy, bw, bh)
                if bbox is not None:
                    parsed_boxes.append(bbox)

            self.current_bboxes = parsed_boxes
            self.bbox_precise = [self._pixels_to_yolo(b) for b in parsed_boxes]
        self._update_overlap_counts()
        self.active_bbox_index = len(self.current_bboxes) - 1 if self.current_bboxes else None
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
        print("  - [V]: Paste Last Bounding Box")
        print("  - [Arrow Keys]: Nudge Most Recent Bounding Box")
        print("  - [T/Y]: Grow/Shrink Most Recent Bounding Box")
        print("  - [G/H]: Widen/Narrow Most Recent Bounding Box")
        print("  - [1-9]: Select Class")
        print("  - [I]: Toggle Class Name Overlay")
        print("-" * 50)
        print(" Navigation & Saving:")
        print("  - [D]: Save & Next Image")
        print("  - [A]: Save & Previous Image")
        print("  - [Q]: Save & Quit")
        print("  - [C]: Force Quit (discards current file's changes)")
        print("-" * 50)
        print(" Workflow:")
        print("  - [F]: Flag / Unflag for Review")
        print("  - [J]: Show All Images")
        print("  - [K]: Show Review-Flagged Images")
        print("  - [N]: Show Overlapping Images")
        print("  - [M]: Show Unlabeled Images")
        print("  - [O]: Toggle View (Dataset / Isolated)")
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
                image_path = self.image_paths[self.img_index]
                img_name = os.path.basename(image_path)
                relative_dir = os.path.relpath(os.path.dirname(image_path), self.dataset_dir)
                if relative_dir == '.' or relative_dir.startswith('..'):
                    relative_dir = ''
                else:
                    relative_dir = relative_dir.replace(os.sep, '/')
                display_name = f"{relative_dir}/{img_name}" if relative_dir else img_name
                try:
                    progress = f"({self.filtered_image_indices.index(self.img_index) + 1}/{len(self.filtered_image_indices)})"
                except ValueError:
                    progress = "(0/0)"

                title = f"Labeler {progress} - {display_name} - MODE: {self.mode.upper()}"
                if self.mode == 'draw':
                    title += f" (Class: {self.classes.get(self.current_class_id, 'N/A')})"
                cv2.setWindowTitle(self.window_name, title)

                self._redraw_ui()
                self._update_magnifier()

                cv2.imshow(self.window_name, self.clone)
                key = cv2.waitKeyEx(1)
                if key == -1:
                    continue

                if key in self.arrow_key_codes['left']:
                    self._move_last_bbox(-self.bbox_nudge_step, 0)
                    continue
                if key in self.arrow_key_codes['right']:
                    self._move_last_bbox(self.bbox_nudge_step, 0)
                    continue
                if key in self.arrow_key_codes['up']:
                    self._move_last_bbox(0, -self.bbox_nudge_step)
                    continue
                if key in self.arrow_key_codes['down']:
                    self._move_last_bbox(0, self.bbox_nudge_step)
                    continue

                key_char = key & 0xFF
                key_lower = chr(key_char).lower() if 0 <= key_char < 256 else ''

                if key_lower in ['q', 'c']:
                    if key_lower != 'c': self._save_current_labels()
                    self.quit_flag = True; break
                elif key_lower in ['d', 's']: # Next
                    self._save_current_labels(); self._navigate(1); break
                elif key_lower == 'a': # Previous
                    self._save_current_labels(); self._navigate(-1); break
                elif key_lower == 'r': # Undo
                    if self.history:
                        prev_boxes, prev_precise = self.history.pop()
                        self.current_bboxes = prev_boxes
                        self.bbox_precise = prev_precise
                    self.active_bbox_index = len(self.current_bboxes) - 1 if self.current_bboxes else None
                    self._update_overlap_counts()
                elif key_lower == 't':
                    self._scale_last_bbox(grow=True)
                elif key_lower == 'y':
                    self._scale_last_bbox(grow=False)
                elif key_lower == 'g':
                    self._scale_last_bbox_width(grow=True)
                elif key_lower == 'h':
                    self._scale_last_bbox_width(grow=False)
                elif key_lower == 'v':
                    if self.clipboard_bbox is None:
                        print("[Info] No bounding box available to paste yet.")
                    else:
                        candidate = self._sanitize_bbox_for_current_image(self.clipboard_bbox)
                        if candidate is None:
                            print("[Warning] Last bounding box does not fit within the current image dimensions.")
                        else:
                            self._ensure_precise_state()
                            self._push_history()
                            self.current_bboxes.append(candidate)
                            self.bbox_precise.append(self._pixels_to_yolo(candidate))
                            self.clipboard_bbox = candidate
                            self.active_bbox_index = len(self.current_bboxes) - 1
                            print("-> Pasted last bounding box.")
                            self._update_overlap_counts()
                elif ord('1') <= key_char <= ord('9'):
                    pressed_key = int(chr(key_char))
                    mapped_class = self.hotkey_to_class.get(pressed_key)
                    if mapped_class is not None:
                        self.current_class_id = mapped_class
                        print(f"Current class set to: {self.classes.get(self.current_class_id, 'N/A')} (ID {self.current_class_id})")
                        if self.mode == 'delete':
                            self.mode = 'draw'
                            print("-> Switched to Draw Mode")
                    else:
                        print(f"[Warning] No class is mapped to number key {pressed_key}.")
                elif key_lower == 'w': self.mode = 'draw'
                elif key_lower == 'e': self.mode = 'delete'
                elif key_lower == 'i':
                    self.show_class_names = not self.show_class_names
                    state = 'ON' if self.show_class_names else 'OFF'
                    print(f"Class name overlay toggled {state}.")
                elif key_lower == 'f': # Flag for review
                    n = os.path.basename(self.image_paths[self.img_index])
                    if n in self.review_list: self.review_list.remove(n)
                    else: self.review_list.add(n)
                elif key_lower == 'x': # Exclude
                    self._isolate_current_image(); break
                elif key_lower == 'j': # Show all images
                    if self.filter_mode != 'all' or not self.filtered_image_indices:
                        previous_index = self.img_index
                        self.filter_mode = 'all'
                        self._apply_filter(announce=True)
                        if self.filtered_image_indices:
                            if previous_index in self.filtered_image_indices:
                                self.img_index = previous_index
                            else:
                                self.img_index = self.filtered_image_indices[0]
                        if self.viewing_isolated:
                            self.last_isolated_index = self.img_index
                        else:
                            self.last_primary_index = self.img_index
                        break
                    else:
                        print("[Info] Already showing all images.")
                elif key_lower == 'k': # Review filter
                    if self.viewing_isolated:
                        print("[Info] Review filter is unavailable in isolated view.")
                        continue
                    previous_mode = self.filter_mode
                    previous_indices = list(getattr(self, 'filtered_image_indices', []))
                    self.filter_mode = 'review'
                    result = self._apply_filter(announce=True, preserve_on_empty=True)
                    if result == 0:
                        self.filter_mode = previous_mode
                        self.filtered_image_indices = previous_indices
                        continue
                    if self.filtered_image_indices:
                        self.img_index = self.filtered_image_indices[0]
                    self.last_primary_index = self.img_index
                    break
                elif key_lower == 'n': # Overlap filter
                    previous_mode = self.filter_mode
                    previous_indices = list(getattr(self, 'filtered_image_indices', []))
                    self.filter_mode = 'overlap'
                    result = self._apply_filter(announce=True, preserve_on_empty=True)
                    if result == 0:
                        self.filter_mode = previous_mode
                        self.filtered_image_indices = previous_indices
                        continue
                    if self.filtered_image_indices:
                        self.img_index = self.filtered_image_indices[0]
                    if self.viewing_isolated:
                        self.last_isolated_index = self.img_index
                    else:
                        self.last_primary_index = self.img_index
                    break
                elif key_lower == 'm': # Unlabeled filter
                    previous_mode = self.filter_mode
                    previous_indices = list(getattr(self, 'filtered_image_indices', []))
                    self.filter_mode = 'unlabeled'
                    result = self._apply_filter(announce=True, preserve_on_empty=True)
                    if result == 0:
                        self.filter_mode = previous_mode
                        self.filtered_image_indices = previous_indices
                        continue
                    if self.filtered_image_indices:
                        self.img_index = self.filtered_image_indices[0]
                    if self.viewing_isolated:
                        self.last_isolated_index = self.img_index
                    else:
                        self.last_primary_index = self.img_index
                    break
                elif key_lower == 'o': # Toggle isolated workspace view
                    self._toggle_isolated_view()
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
