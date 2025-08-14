import cv2
import os
import glob
import yaml
import argparse
import numpy as np
from collections import Counter

# ==============================================================================
# Initial Configuration
# ==============================================================================
# Optional: Set a default dataset and starting image to streamline startup.
# Example: INIT_DATASET_DIR = "datasets/my_project_v1"
# Example: INIT_IMAGE_NAME = "frame_0123.jpg"
INIT_DATASET_DIR = None
INIT_IMAGE_NAME = None
# ==============================================================================

# --- Global variables for sharing state between mouse callback and main loop ---
ref_point, current_bboxes, history, previous_image_new_bboxes = [], [], [], []
drawing, deletion_mode = False, False
current_class_id = 0
clone, img_orig, display_img = None, None, None
h_orig, w_orig, ratio = 0, 0, 1.0
CLASSES, COLORS = {}, {}

# --- Display constants ---
DISPLAY_WIDTH, MAGNIFIED_SIZE, MAGNIFIED_REGION = 1280, 480, 100


def pixels_to_yolo(pixel_bbox, img_width, img_height):
    """Converts a bounding box from pixel coordinates to YOLO format."""
    class_id, x1, y1, x2, y2 = pixel_bbox
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return class_id, x_center / img_width, y_center / img_height, width / img_width, height / img_height


def draw_dotted_rectangle(img, pt1, pt2, color, thickness, gap=10):
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


def print_controls():
    """Prints the control keys to the console for the user."""
    print("\n" + "="*40)
    print(" " * 14 + "Manual Labeler Controls")
    print("="*40)
    print("  - [w]: Drawing Mode (Default) | [e]: Deletion Mode")
    print("  - [1-9]: Select Class         | [r]: Undo Last Action")
    print("  - [s]: Save and Next          | [a]: Save and Previous")
    print("  - [v]: Paste Boxes from Previous Image")
    print("  - [q]: Save and Quit          | [c]: Force Quit without Saving")
    print("="*40)


def redraw_boxes(image):
    """Redraws all current bounding boxes onto the display image."""
    for bbox in current_bboxes:
        class_id, x1, y1, x2, y2 = bbox
        color = COLORS.get(class_id, (255, 255, 255)) # Default to white if class not found
        disp_x1, disp_y1 = int(x1 * ratio), int(y1 * ratio)
        disp_x2, disp_y2 = int(x2 * ratio), int(y2 * ratio)
        cv2.rectangle(image, (disp_x1, disp_y1), (disp_x2, disp_y2), color, 2)
        cv2.putText(image, f"{CLASSES.get(class_id, 'Unknown')}", (disp_x1, disp_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def click_and_draw_box(event, x, y, flags, param):
    """Mouse callback function to handle drawing and deleting boxes."""
    global ref_point, current_bboxes, drawing, clone, deletion_mode, history

    orig_x, orig_y = int(x / ratio), int(y / ratio) # Scale coordinates to original image size

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            ref_point, drawing = [(orig_x, orig_y)], True
        else:
            drawing = False
            ref_point.append((orig_x, orig_y))
            history.append(current_bboxes.copy()) # Save state for undo

            if deletion_mode:
                del_x1, del_y1 = min(ref_point[0][0], ref_point[1][0]), min(ref_point[0][1], ref_point[1][1])
                del_x2, del_y2 = max(ref_point[0][0], ref_point[1][0]), max(ref_point[0][1], ref_point[1][1])
                
                bboxes_to_keep = [b for b in current_bboxes if not (del_x1 < (b[1] + b[3]) / 2 < del_x2 and del_y1 < (b[2] + b[4]) / 2 < del_y2)]
                removed_count = len(current_bboxes) - len(bboxes_to_keep)
                if removed_count > 0:
                    print(f"-> Removed {removed_count} boxes.")
                    current_bboxes = bboxes_to_keep
            else:
                x1, y1 = ref_point[0]
                x2, y2 = ref_point[1]
                current_bboxes.append((current_class_id, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            
            ref_point = []
            clone = display_img.copy()
            redraw_boxes(clone)

    elif event == cv2.EVENT_MOUSEMOVE:
        clone = display_img.copy()
        redraw_boxes(clone)
        h_disp, w_disp = clone.shape[:2]

        # Draw the box-in-progress or the deletion rectangle
        if drawing and ref_point:
            start_disp = (int(ref_point[0][0] * ratio), int(ref_point[0][1] * ratio))
            if deletion_mode:
                draw_dotted_rectangle(clone, start_disp, (x, y), (0, 0, 255), 2)
            else:
                color = COLORS.get(current_class_id, (0, 255, 255))
                cv2.rectangle(clone, start_disp, (x, y), color, 2)
                cv2.putText(clone, CLASSES.get(current_class_id, "Unknown"), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw crosshairs and magnified view
        cross_color = (0, 0, 255) if deletion_mode else COLORS.get(current_class_id, (0, 255, 255))
        cv2.line(clone, (0, y), (w_disp, y), cross_color, 1)
        cv2.line(clone, (x, 0), (x, h_disp), cross_color, 1)
        
        x_s, y_s = max(0, x - MAGNIFIED_REGION), max(0, y - MAGNIFIED_REGION)
        x_e, y_e = min(w_disp, x + MAGNIFIED_REGION), min(h_disp, y + MAGNIFIED_REGION)
        mag_region = clone[y_s:y_e, x_s:x_e]
        
        if mag_region.size > 0:
            cv2.imshow("Magnified View", cv2.resize(mag_region, (MAGNIFIED_SIZE, MAGNIFIED_SIZE), interpolation=cv2.INTER_NEAREST))


def save_labels(label_path, bboxes, w, h):
    """Saves the list of bounding boxes to a file in YOLO format."""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as f:
        if not bboxes:
            print(f"  -> Save: No labels, saving empty file.")
            return
        for bbox in bboxes:
            yolo_bbox = pixels_to_yolo(bbox, w, h)
            f.write(f"{yolo_bbox[0]} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f} {yolo_bbox[4]:.6f}\n")
            
    summary = ", ".join([f"{CLASSES.get(c, 'N/A')}: {cnt}" for c, cnt in sorted(Counter(b[0] for b in bboxes).items())])
    print(f"  -> Saved: '{os.path.basename(label_path)}' ({summary})")


def process_newly_added_boxes(initial_bboxes, final_bboxes):
    """Identifies newly added boxes to be available for pasting on the next image."""
    global previous_image_new_bboxes
    newly_added = list(set(final_bboxes) - set(initial_bboxes))
    if newly_added:
        print(f"  -> Copied {len(newly_added)} new boxes for next image. (Paste with 'v')")
        previous_image_new_bboxes = newly_added
    else:
        previous_image_new_bboxes = []


def main(config, args):
    """Main logic for the Manual Labeler."""
    global CLASSES, COLORS, ref_point, current_bboxes, current_class_id, drawing, deletion_mode, history, clone, img_orig, display_img, h_orig, w_orig, ratio
    
    CLASSES = config['model_configurations']['classes']
    COLORS = {c: ((c * 55 + 50) % 256, (c * 95 + 100) % 256, (c * 135 + 150) % 256) for c in CLASSES.keys()}
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir_rel = args.dataset if args.dataset is not None else INIT_DATASET_DIR if INIT_DATASET_DIR is not None else config['dataset_paths']['initial_labeled_dataset']
    dataset_dir = os.path.join(project_root, dataset_dir_rel)
    start_from_image = args.start_image if args.start_image is not None else INIT_IMAGE_NAME
    
    print("\n" + "="*50)
    print("Starting Manual Labeler.")
    print("="*50)
    print(f"  - Target Dataset: {dataset_dir}")
    if start_from_image:
        print(f"  - Starting Image: {start_from_image}")
    print("="*50)

    images_base_dir = os.path.join(dataset_dir, 'images')
    labels_base_dir = os.path.join(dataset_dir, 'labels')
    os.makedirs(labels_base_dir, exist_ok=True)
    
    all_image_paths = sorted([p for ext in config['workflow_parameters']['image_format'].split(',') for p in glob.glob(os.path.join(images_base_dir, '**', f'*.{ext}'), recursive=True)])
    if not all_image_paths:
        print(f"Error: No images found in '{images_base_dir}'.")
        return
        
    print(f"Found {len(all_image_paths)} total images.")
    print_controls()
    
    start_mode = input("Select start mode (1: From first unlabeled, 2: From specific image, 3: From review_list.txt): ")
    img_index, image_paths, is_review_mode = 0, all_image_paths, False

    if start_mode == '3':
        is_review_mode = True
        review_list_path = os.path.join(dataset_dir, 'review_list.txt')
        if not os.path.exists(review_list_path):
            print(f"Error: '{review_list_path}' not found.")
            return
        with open(review_list_path, 'r') as f:
            review_files = {line.strip() for line in f if line.strip()}
        image_paths = [p for p in all_image_paths if os.path.basename(p) in review_files]
        if not image_paths:
            print("No images in the dataset match the review list.")
            return
        print(f"\nReview Mode: Reviewing {len(image_paths)} images.")
    elif start_mode == '2':
        start_img_name = start_from_image or input("Enter starting image name (with extension): ")
        try:
            img_index = [os.path.basename(p) for p in image_paths].index(start_img_name)
            print(f"\nStarting from '{start_img_name}'.")
        except ValueError:
            print(f"Error: Could not find image '{start_img_name}'.")
            return
    else:
        print("\nNormal Mode: Starting from the first image without a label file.")
        for i, path in enumerate(image_paths):
            label_file = os.path.splitext(path.replace('images', 'labels', 1))[0] + '.txt'
            if not os.path.exists(label_file):
                img_index = i
                break
        else:
            print("All images have label files. Starting from the first image.")
            img_index = 0
    
    if not image_paths:
        print("No images to process.")
        return
        
    cv2.namedWindow("Manual Labeler")
    cv2.namedWindow("Magnified View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Magnified View", MAGNIFIED_SIZE, MAGNIFIED_SIZE)

    quit_flag = False
    while 0 <= img_index < len(image_paths) and not quit_flag:
        image_path = image_paths[img_index]
        image_name = os.path.basename(image_path)
        label_path = os.path.splitext(image_path.replace('images', 'labels', 1))[0] + '.txt'
        
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            print(f"Warning: Could not load '{image_name}'. Skipping.")
            img_index += 1
            continue
            
        h_orig, w_orig = img_orig.shape[:2]
        ratio = DISPLAY_WIDTH / w_orig
        display_img = cv2.resize(img_orig, (DISPLAY_WIDTH, int(h_orig * ratio)))
        
        clone = display_img.copy()
        current_bboxes, ref_point, drawing, history = [], [], False, []
        
        # Load existing labels if they exist
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cid, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                        x1 = int((xc - bw / 2) * w_orig)
                        y1 = int((yc - bh / 2) * h_orig)
                        x2 = int((xc + bw / 2) * w_orig)
                        y2 = int((yc + bh / 2) * h_orig)
                        current_bboxes.append((cid, x1, y1, x2, y2))
            except Exception as e:
                print(f"Warning: Error reading '{os.path.basename(label_path)}': {e}")
                
        initial_bboxes_for_image = current_bboxes.copy()
        redraw_boxes(clone)
        cv2.setMouseCallback("Manual Labeler", click_and_draw_box)

        # Main event loop for a single image
        while True:
            mode_text = "Area Delete" if deletion_mode else "Draw Box"
            review_text = "[Review Mode] " if is_review_mode else ""
            title = f"{review_text}[{img_index+1}/{len(image_paths)}] {image_name} | Class: {CLASSES.get(current_class_id,'N/A')} | Mode: {mode_text}"
            cv2.setWindowTitle("Manual Labeler", title)
            cv2.imshow("Manual Labeler", clone)
            
            key = cv2.waitKey(1) & 0xFF

            # 'q', 's', 'a', 'c': Quit, Save, or Navigate
            if key in [ord('q'), ord('s'), ord('a'), ord('c')]:
                process_newly_added_boxes(initial_bboxes_for_image, current_bboxes)
                if key != ord('c'):
                    save_labels(label_path, current_bboxes, w_orig, h_orig)
                if is_review_mode and 'review_files' in locals() and image_name in review_files:
                    review_files.remove(image_name)
                    if key == 's':
                        print(f"-> Reviewed '{image_name}'. {len(review_files)} files remaining.")
                if key == ord('q'):
                    quit_flag = True
                    print("Quitting.")
                    break
                elif key == ord('c'):
                    quit_flag = True
                    print("Force quitting without saving.")
                    break
                elif key == ord('s'):
                    img_index += 1
                    break
                elif key == ord('a'):
                    if img_index > 0:
                        img_index -= 1
                        print("-> Going to previous image.")
                        break
                    else:
                        print("-> Already at the first image.")
            
            # 'r': Undo last action
            elif key == ord('r'):
                if drawing:
                    drawing, ref_point = False, []
                    print("-> Canceled drawing.")
                elif history:
                    current_bboxes = history.pop()
                    print("-> Undid last action.")
                else:
                    print("-> Nothing to undo.")
                clone = display_img.copy()
                redraw_boxes(clone)

            # 'v': Paste boxes from previous image
            elif key == ord('v'):
                if previous_image_new_bboxes:
                    history.append(current_bboxes.copy())
                    current_bboxes.extend(previous_image_new_bboxes)
                    print(f"-> Pasted {len(previous_image_new_bboxes)} boxes.")
                    clone = display_img.copy()
                    redraw_boxes(clone)
                else:
                    print("-> No new boxes from previous image to paste.")

            # '1'-'9': Change class and auto-switch from deletion mode
            elif ord('1') <= key <= ord('9'):
                new_class_id = int(chr(key)) - 1
                if new_class_id in CLASSES:
                    current_class_id = new_class_id
                    print(f"-> Class changed to: {CLASSES[current_class_id]}")
                    
                    if deletion_mode:
                        deletion_mode = False
                        drawing = False
                        ref_point = []
                        print(f"\n>> Switched to Draw Box Mode <<")

            # 'e', 'w': Toggle between drawing and deletion modes
            elif key in [ord('e'), ord('w')]:
                new_mode = (key == ord('e'))
                if deletion_mode != new_mode:
                    deletion_mode = new_mode
                    drawing, ref_point = False, []
                    print(f"\n>> Switched to {'Area Delete' if new_mode else 'Draw Box'} Mode <<")

    # After loop, handle review list persistence
    if is_review_mode and 'review_files' in locals():
        review_list_path = os.path.join(dataset_dir, 'review_list.txt')
        if not review_files:
            if os.path.exists(review_list_path):
                os.remove(review_list_path)
                print(f"\nAll items reviewed. Deleted '{review_list_path}'.")
        else:
            with open(review_list_path, 'w') as f:
                f.write('\n'.join(sorted(list(review_files))))
            print(f"\nProgress saved. {len(review_files)} items remaining in review list.")
            
    cv2.destroyAllWindows()
    print("\nManual Labeler closed.")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: _config.yaml not found. Check if the file exists in the project root.")
        exit()
        
    parser = argparse.ArgumentParser(description="A tool for manually labeling image datasets in YOLO format.")
    parser.add_argument('--dataset', type=str, default=None, help="Relative path to the dataset directory to be labeled.")
    parser.add_argument('--start_image', type=str, default=None, help="Filename of the image to start labeling from.")
    args = parser.parse_args()
    
    main(config, args)