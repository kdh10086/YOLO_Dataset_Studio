import os
import sys
import platform
import threading
import yaml
import argparse
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 플랫폼(OS)에 따른 키보드 입력 라이브러리 임포트
if platform.system() == "Windows":
    import msvcrt
else:
    import tty
    import termios
    import select

# ==============================================================================
# 초기 실행 설정 (Initiation Settings)
#
# - Code Runner 또는 IDE에서 '직접 실행' 시 이 부분을 수정하여 사용하세요.
# - 터미널에서 인자를 직접 지정하면 이 설정은 무시됩니다.
# - 값을 None으로 두면 _config.yaml의 기본 설정을 따릅니다.
# ==============================================================================
INIT_DATASET_DIR = None   # 예시: 'datasets/sample_dataset'
INIT_EPOCHS = None        # 예시: 50
INIT_BATCH_SIZE = None    # 예시: 16
INIT_IMG_SIZE = None      # 예시: 640
# ==============================================================================

# --- 학습 중단 처리를 위한 전역 변수 및 클래스 ---
stop_training_flag = False

class TQDMProgressBar:
    """YOLO 학습 진행률을 TQDM으로 보여주는 콜백 클래스."""
    def __init__(self):
        self.pbar = None
    def on_train_start(self, trainer):
        self.pbar = tqdm(total=trainer.epochs, desc="🚀 Overall Training Progress", unit="epoch")
    def on_epoch_end(self, trainer):
        metrics = trainer.metrics
        metrics_str = f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}, BoxLoss: {metrics.get('val/box_loss', 0):.4f}"
        self.pbar.set_description(f"Epoch {trainer.epoch}/{trainer.epochs} ({metrics_str})")
        self.pbar.update(1)
    def on_train_end(self, trainer):
        if self.pbar: self.pbar.close()

# --- 헬퍼 함수 ---
def check_for_quit_key():
    """'q' 키 입력을 감지하여 학습 중단 플래그를 설정하는 스레드 함수."""
    global stop_training_flag
    if platform.system() == "Windows":
        while not stop_training_flag:
            if msvcrt.kbhit() and msvcrt.getch().decode(errors='ignore').lower() == 'q':
                stop_training_flag = True; break
    else: # Linux/macOS
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not stop_training_flag:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    if sys.stdin.read(1).lower() == 'q':
                        stop_training_flag = True; break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    if stop_training_flag:
        print("\n'q' 키 입력 감지! 현재 에포크 완료 후 학습을 중단합니다...")

def display_training_results(results_dir):
    """학습 완료 후 결과 이미지를 화면에 표시합니다."""
    print("\n[결과 시각화] 학습 결과 그래프를 출력합니다...")
    try:
        results_png = os.path.join(results_dir, 'results.png')
        if os.path.exists(results_png):
            Image.open(results_png).show(title="Training & Validation Metrics")
        
        confusion_matrix_png = os.path.join(results_dir, 'confusion_matrix.png')
        if os.path.exists(confusion_matrix_png):
            Image.open(confusion_matrix_png).show(title="Confusion Matrix")
    except Exception as e:
        print(f"오류: 그래프 출력 중 오류가 발생했습니다: {e}")

# --- 메인 학습 함수 ---
def train_model(config, args):
    """YOLO 모델 학습을 설정하고 실행합니다."""
    global stop_training_flag
    
    # 1. 설정값 결정 (3단계 우선순위 적용)
    project_root = os.path.dirname(os.path.abspath(__file__))

    dataset_dir_relative = args.dataset if args.dataset is not None else \
                           INIT_DATASET_DIR if INIT_DATASET_DIR is not None else \
                           config['datasets']['sample']
    dataset_dir = os.path.join(project_root, dataset_dir_relative)
    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')

    model_name = config['model_name']
    h_params = config['hyperparameters']
    model_specific_params = h_params['models'].get(model_name.split('.')[0], h_params['models']['default'])

    epochs = args.epochs if args.epochs is not None else INIT_EPOCHS if INIT_EPOCHS is not None else h_params['epochs']
    batch_size = args.batch if args.batch is not None else INIT_BATCH_SIZE if INIT_BATCH_SIZE is not None else model_specific_params['batch_size']
    img_size = args.imgsz if args.imgsz is not None else INIT_IMG_SIZE if INIT_IMG_SIZE is not None else model_specific_params['img_size']
    patience = h_params['patience']
    
    # --- 설정 확인 및 유효성 검사 ---
    print("\n" + "="*50); print("YOLO 모델 학습을 시작합니다."); print("="*50)
    print(f"  - 학습 대상 데이터셋: {dataset_dir}")
    print(f"  - 모델: {model_name}")
    print(f"  - Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {img_size}")
    print("="*50)

    if not os.path.exists(data_yaml_path):
        print(f"오류: data.yaml 파일을 찾을 수 없습니다! 경로: {data_yaml_path}")
        print("3_split_dataset.py를 먼저 실행하여 data.yaml을 생성했는지 확인하세요.")
        return

    # 2. 키보드 리스너 스레드 시작
    listener_thread = threading.Thread(target=check_for_quit_key, daemon=True)
    listener_thread.start()
    print("학습 중 'q'를 누르면 안전하게 종료할 수 있습니다.")

    # 3. 모델 및 콜백 설정
    model = YOLO(f"{model_name}.pt")
    
    progress_callback = TQDMProgressBar()
    model.add_callback("on_train_start", progress_callback.on_train_start)
    model.add_callback("on_epoch_end", progress_callback.on_epoch_end)
    model.add_callback("on_train_end", progress_callback.on_train_end)
    
    def check_quit_callback(trainer):
        if stop_training_flag: trainer.stop = True
    model.add_callback("on_batch_end", check_quit_callback)

    # 4. 학습 실행
    results = None
    training_successful = False
    try:
        run_name = f"{model_name}_on_{os.path.basename(dataset_dir_relative)}"
        
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            patience=patience,
            batch=batch_size,
            imgsz=img_size,
            project=os.path.join(project_root, 'runs/train'),
            name=run_name,
            exist_ok=True, # 동일 이름의 실험에 덮어쓰기 허용
            optimizer='auto'
        )
        
        if not stop_training_flag:
            print("\n🎉 학습이 성공적으로 완료되었습니다!")
            training_successful = True
        else:
            print("\n🛑 사용자의 요청으로 학습이 중단되었습니다.")
        
        if results:
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            if os.path.exists(best_model_path):
                print(f"✅ 최종 모델 저장 경로:\n   {best_model_path}")

    except Exception as e:
        print(f"🔥 학습 중 오류가 발생했습니다: {e}")
    finally:
        stop_training_flag = True # 스레드가 살아있을 경우를 대비해 확실히 종료
    
    # 5. 결과 시각화
    if results and training_successful:
        display_training_results(results.save_dir)

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("오류: _config.yaml 파일을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인하세요.")
        exit()

    parser = argparse.ArgumentParser(description="YOLO 모델을 학습시킵니다.")
    parser.add_argument('--dataset', type=str, default=None, help="학습시킬 데이터셋의 상대 경로. (예: 'datasets/sample_dataset')")
    parser.add_argument('--epochs', type=int, default=None, help="Epoch 횟수를 지정합니다.")
    parser.add_argument('--batch', type=int, default=None, help="배치 사이즈를 지정합니다.")
    parser.add_argument('--imgsz', type=int, default=None, help="학습 이미지 사이즈를 지정합니다.")
    args = parser.parse_args()

    train_model(config, args)