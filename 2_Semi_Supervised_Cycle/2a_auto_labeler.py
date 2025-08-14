import os
import glob
import yaml
import argparse
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ==============================================================================
# Initiation Settings
# ==============================================================================
INIT_DATASET_DIR = None
INIT_TEACHER_MODEL_PATH = None
INIT_CONF_THRESHOLD = None
INIT_BATCH_SIZE = None
# ==============================================================================

def auto_label_dataset(config, args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    teacher_cfg = config['model_configurations']['teacher_model_config']
    
    dataset_dir_rel = args.dataset or INIT_DATASET_DIR or config['dataset_paths']['unlabeled_pool_dir']
    dataset_dir = os.path.join(project_root, dataset_dir_rel)
    
    teacher_model_rel = args.weights or INIT_TEACHER_MODEL_PATH or config['model_configurations']['semi_supervised_weights']
    teacher_model_path = os.path.join(project_root, teacher_model_rel)
    
    conf_threshold = args.conf or INIT_CONF_THRESHOLD or config['workflow_parameters']['auto_label_confidence_threshold']

    model_name = teacher_cfg['model_name']
    h_params = teacher_cfg['hyperparameters']
    default_batch_size = h_params['models'].get(model_name, h_params['models']['default'])['batch_size']
    batch_size = args.batch or INIT_BATCH_SIZE or default_batch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "="*50); print("Starting Auto-Labeling."); print("="*50)
    print(f"  - Target Dataset: {dataset_dir}"); print(f"  - Teacher Model: {teacher_model_path}")
    print(f"  - Confidence Threshold: {conf_threshold}"); print(f"  - Batch Size: {batch_size}")
    print(f"  - Device: {device.upper()}"); print("="*50)

    images_dir = os.path.join(dataset_dir, 'images')
    if not os.path.isdir(images_dir): print(f"Error: 'images' directory not found: {images_dir}"); return
    if not os.path.exists(teacher_model_path): print(f"Error: Teacher model not found: {teacher_model_path}"); return

    try:
        model = YOLO(teacher_model_path); model.to(device)
    except Exception as e: print(f"Error: Failed to load model: {e}"); return

    labels_dir = os.path.join(dataset_dir, 'labels'); os.makedirs(labels_dir, exist_ok=True)
    image_paths = [p for fmt in config['workflow_parameters']['image_format'].split(',') for p in glob.glob(os.path.join(images_dir, f'*.{fmt}'))]
    if not image_paths: print(f"Error: No images to label in '{images_dir}'"); return
        
    print(f"Starting auto-labeling for {len(image_paths)} images.")

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Auto-labeling in progress"):
        batch_paths = image_paths[i:i + batch_size]
        try:
            results = model(batch_paths, conf=conf_threshold, verbose=False)
            for res in results:
                label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(res.path))[0] + '.txt')
                with open(label_path, 'w') as f:
                    for box in res.boxes:
                        class_id, (x, y, w, h) = int(box.cls), box.xywhn[0]
                        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            print(f"\nError: An error occurred during inference (Image: {batch_paths[0]}...)\n - {e}"); continue

    print("\nAuto-labeling process completed!"); print(f"Label files have been saved in the '{labels_dir}' folder.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f: config = yaml.safe_load(f)
    except FileNotFoundError: print("Error: _config.yaml not found."); exit()
    parser = argparse.ArgumentParser(description="Automatically labels an image dataset using a teacher model.")
    parser.add_argument('--dataset', type=str, default=None, help="Relative path to the dataset to be auto-labeled.")
    parser.add_unlabeled_pool_dir('--weights', type=str, default=None, help="Relative path to the teacher model weights (.pt) file.")
    parser.add_argument('--conf', type=float, default=None, help="Confidence threshold for object detection.")
    parser.add_argument('--batch', type=int, default=None, help="Batch size for inference.")
    args = parser.parse_args()
    auto_label_dataset(config, args)