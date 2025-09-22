import os
import sys
import threading
import yaml
from ultralytics import YOLO
from tqdm import tqdm

# POSIX-specific imports for non-blocking keyboard input
try:
    import tty, termios, select
except ImportError:
    # This will fail on Windows, which is expected.
    # The graceful stop feature will be disabled.
    pass

stop_training_flag = False

class TQDMProgressBar:
    """A TQDM progress bar callback for YOLO training."""
    def __init__(self):
        self.pbar = None

    def on_train_start(self, trainer):
        self.pbar = tqdm(total=trainer.epochs, desc="Overall Training Progress", unit="epoch")

    def on_epoch_end(self, trainer):
        metrics = trainer.metrics
        metrics_str = f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}, BoxLoss: {metrics.get('val/box_loss', 0):.4f}"
        self.pbar.set_description(f"Epoch {trainer.epoch+1}/{trainer.epochs} ({metrics_str})")
        self.pbar.update(1)

    def on_train_end(self, trainer):
        if self.pbar:
            self.pbar.close()

def _check_for_quit_key():
    """
    Thread function to detect 'q' key press on POSIX systems (Linux/macOS)
    and set the stop flag. This will not work on Windows.
    """
    global stop_training_flag
    # This feature is only supported on systems with TTY.
    if 'termios' not in sys.modules or not sys.stdin.isatty():
        return

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not stop_training_flag:
            # Check for input with a timeout
            if select.select([sys.stdin], [], [], 0.1)[0]:
                if sys.stdin.read(1).lower() == 'q':
                    stop_training_flag = True
                    break
    except (termios.error, AttributeError):
        # Gracefully fail if termios functions fail (e.g., in non-interactive shells)
        pass
    finally:
        # Always try to restore the terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    if stop_training_flag:
        print("\n'q' key detected! Training will stop after the current epoch.")

def train_yolo_model(dataset_path, model_config_name, role, run_name, global_config, exist_ok=False):
    """
    A unified function to train any YOLO model based on configuration.

    Args:
        dataset_path (str): Absolute path to the dataset directory (containing data.yaml).
        model_config_name (str): The key for the model config in models_config.yaml.
        role (str): The role of the model, e.g., 'teacher' or 'student'.
        run_name (str): The dynamic name for the training run.
        global_config (dict): The loaded models_config.yaml file.
        exist_ok (bool): If True, overwrites existing training results.
    """
    global stop_training_flag
    stop_training_flag = False

    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        print(f"[Error] data.yaml not found in '{dataset_path}'. Please split the dataset first.")
        return

    try:
        model_cfg = global_config['model_configurations'][model_config_name]
        h_params = model_cfg['hyperparameters']
        model_name = model_cfg['model_name']
        model_specific_params = h_params['models'].get(model_name, h_params['models']['default'])

        epochs = h_params['epochs']
        patience = h_params['patience']
        batch_size = model_specific_params['batch_size']
        img_size = model_specific_params['img_size']
    except KeyError as e:
        print(f"[Error] Configuration key missing in models_config.yaml: {e}")
        return

    print("\n" + "="*50)
    print(f"Starting YOLO Model Training: {role.upper()}")
    print("="*50)
    print(f"  - Run Name: {run_name}")
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Model: {model_name}, Epochs: {epochs}, Batch: {batch_size}, ImgSize: {img_size}")
    print("="*50)

    listener_thread = threading.Thread(target=_check_for_quit_key, daemon=True)
    listener_thread.start()
    print("Press 'q' during training to stop gracefully after the current epoch.")

    try:
        model = YOLO(f"{model_name}.pt")
        progress_callback = TQDMProgressBar()
        model.add_callback("on_train_start", progress_callback.on_train_start)
        model.add_callback("on_epoch_end", progress_callback.on_epoch_end)
        model.add_callback("on_train_end", progress_callback.on_train_end)
        model.add_callback("on_batch_end", lambda trainer: setattr(trainer, 'stop', True) if stop_training_flag else None)

        project_dir = os.path.join(os.getcwd(), 'runs', 'train', role)

        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            patience=patience,
            batch=batch_size,
            imgsz=img_size,
            project=project_dir,
            name=run_name,
            exist_ok=exist_ok,
            optimizer='auto'
        )

        if not stop_training_flag:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining was stopped by the user.")

        if results and hasattr(results, 'save_dir'):
            final_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
            if os.path.exists(final_model_path):
                print(f"Final model saved at:\n   {final_model_path}")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
    finally:
        stop_training_flag = True