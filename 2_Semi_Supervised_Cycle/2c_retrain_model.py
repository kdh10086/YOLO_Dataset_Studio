import os
import sys
import platform
import threading
import yaml
import argparse
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

# Import platform-specific libraries for keyboard input
if platform.system() == "Windows": import msvcrt
else: import tty, termios, select

# ==============================================================================
# Initiation Settings
# ==============================================================================
# These global variables define default training parameters. They are used as
# fallbacks if the corresponding settings are not provided via command-line
# arguments or in the _config.yaml file.

# Default relative path to the dataset directory.
INIT_DATASET_DIR = None

# Default number of epochs for training.
INIT_EPOCHS = None

# Default batch size.
INIT_BATCH_SIZE = None

# Default image size (e.g., 640 for 640x640 pixels).
INIT_IMG_SIZE = None

# Default setting for overwriting an existing experiment directory.
# If False, Ultralytics creates a new directory (e.g., 'yolov8n_result2')
# instead of overwriting the existing one.
INIT_EXIST_OK = False
# ==============================================================================

stop_training_flag = False

class TQDMProgressBar:
    """A TQDM progress bar callback for YOLO training."""
    def __init__(self): self.pbar = None
    def on_train_start(self, trainer): self.pbar = tqdm(total=trainer.epochs, desc="Overall Training Progress", unit="epoch")
    def on_epoch_end(self, trainer):
        metrics = trainer.metrics
        metrics_str = f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}, BoxLoss: {metrics.get('val/box_loss', 0):.4f}"
        self.pbar.set_description(f"Epoch {trainer.epoch+1}/{trainer.epochs} ({metrics_str})")
        self.pbar.update(1)
    def on_train_end(self, trainer):
        if self.pbar: self.pbar.close()

def check_for_quit_key():
    """Thread function to detect 'q' key press and set the stop flag."""
    global stop_training_flag
    if platform.system() == "Windows":
        while not stop_training_flag:
            if msvcrt.kbhit() and msvcrt.getch().decode(errors='ignore').lower() == 'q':
                stop_training_flag = True
                break
    else:  # Linux/macOS
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not stop_training_flag:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    if sys.stdin.read(1).lower() == 'q':
                        stop_training_flag = True
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    if stop_training_flag:
        print("\n'q' key detected! Training will stop after the current epoch.")

def display_training_results(results_dir):
    """Displays training result images after completion."""
    print("\n[Result Visualization] Displaying training result graphs...")
    try:
        for f in ['results.png', 'confusion_matrix.png']:
            img_path = os.path.join(results_dir, f)
            if os.path.exists(img_path):
                Image.open(img_path).show()
    except Exception as e:
        print(f"Error: An error occurred while displaying graphs: {e}")

def train_model(config, args):
    """Sets up and runs the YOLO model training."""
    global stop_training_flag
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_cfg = config['model_configurations']['teacher_model_config']
    h_params = model_cfg['hyperparameters']
    
    dataset_dir_rel = args.dataset or INIT_DATASET_DIR or config['dataset_paths']['merged_dataset_for_retrain']
    dataset_dir = os.path.join(project_root, dataset_dir_rel)
    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
    
    model_name = model_cfg['model_name']
    model_specific_params = h_params['models'].get(model_name, h_params['models']['default'])
    
    epochs = args.epochs or INIT_EPOCHS or h_params['epochs']
    batch_size = args.batch or INIT_BATCH_SIZE or model_specific_params['batch_size']
    img_size = args.imgsz or INIT_IMG_SIZE or model_specific_params['img_size']
    patience = h_params['patience']
    
    print("\n" + "="*50); print("Starting YOLO Model Retraining. (Phase 2: Retraining Teacher)"); print("="*50)
    print(f"  - Target Dataset: {dataset_dir}"); print(f"  - Model: {model_name}")
    print(f"  - Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {img_size}"); print("="*50)

    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml not found! Path: {data_yaml_path}"); return

    listener_thread = threading.Thread(target=check_for_quit_key, daemon=True); listener_thread.start()
    print("Press 'q' during training to stop gracefully after the current epoch.")

    model = YOLO(f"{model_name}.pt")
    progress_callback = TQDMProgressBar()
    model.add_callback("on_train_start", progress_callback.on_train_start)
    model.add_callback("on_epoch_end", progress_callback.on_epoch_end)
    model.add_callback("on_train_end", progress_callback.on_train_end)
    model.add_callback("on_batch_end", lambda trainer: setattr(trainer, 'stop', True) if stop_training_flag else None)

    try:
        project_dir = os.path.join(project_root, 'runs/train')
        run_name = f"{model_name}_result"
        
        # Overwrite if --exist_ok flag is passed, otherwise use INIT setting
        exist_ok = args.exist_ok or (INIT_EXIST_OK is True)
        
        results = model.train(data=data_yaml_path, epochs=epochs, patience=patience, batch=batch_size,
                              imgsz=img_size, project=project_dir,
                              name=run_name, exist_ok=exist_ok, optimizer='auto')
        
        if not stop_training_flag:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining was stopped by the user.")
        
        if results and hasattr(results, 'save_dir'):
            final_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
            if os.path.exists(final_model_path):
                print(f"Final model saved at:\n   {final_model_path}")
                if not stop_training_flag:
                    display_training_results(results.save_dir)

    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        stop_training_flag = True # Ensure thread exits

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: _config.yaml not found."); exit()

    parser = argparse.ArgumentParser(description="Retrains the YOLO Teacher model.")
    parser.add_argument('--dataset', type=str, default=None, help="Relative path to the training dataset.")
    parser.add_argument('--epochs', type=int, default=None, help="Number of training epochs.")
    parser.add_argument('--batch', type=int, default=None, help="Batch size.")
    parser.add_argument('--imgsz', type=int, default=None, help="Training image size.")
    parser.add_argument('--exist_ok', action='store_true', help="If set, overwrites the existing results directory.")
    args = parser.parse_args()
    train_model(config, args)