import os
import glob
import yaml
import argparse
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ==============================================================================
# 초기 실행 설정 (Initiation Settings)
#
# - Code Runner 또는 IDE에서 '직접 실행' 시 이 부분을 수정하여 사용하세요.
# - 터미널에서 인자를 직접 지정하면 이 설정은 무시됩니다.
# - 값을 None으로 두면 _config.yaml의 기본 설정을 따릅니다.
# ==============================================================================
INIT_DATASET_DIR = None         # 예시: 'datasets/track_dataset_0811'
INIT_TEACHER_MODEL_PATH = None  # 예시: 'runs/train/yolov10s_result/weights/best.pt'
INIT_CONF_THRESHOLD = None      # 예시: 0.25
INIT_BATCH_SIZE = None          # 예시: 32
# ==============================================================================


def auto_label_dataset(config, args):
    """
    선생님 모델을 사용하여 데이터셋의 이미지들을 자동으로 라벨링합니다.
    """
    # 1. 설정값 결정 (3단계 우선순위 적용)
    project_root = os.path.dirname(os.path.abspath(__file__))

    dataset_dir_relative = args.dataset if args.dataset is not None else \
                           INIT_DATASET_DIR if INIT_DATASET_DIR is not None else \
                           config['datasets']['raw']
    dataset_dir = os.path.join(project_root, dataset_dir_relative)
    
    teacher_model_relative = args.weights if args.weights is not None else \
                             INIT_TEACHER_MODEL_PATH if INIT_TEACHER_MODEL_PATH is not None else \
                             config['teacher_model_path']
    teacher_model_path = os.path.join(project_root, teacher_model_relative)
    
    conf_threshold = args.conf if args.conf is not None else \
                     INIT_CONF_THRESHOLD if INIT_CONF_THRESHOLD is not None else \
                     config['confidence_threshold']

    # YAML에 정의된 모델별 기본 배치 사이즈를 가져옴
    model_name = config['model_name']
    h_params = config['hyperparameters']
    default_batch_size = h_params['models'].get(model_name, h_params['models']['default'])['batch_size']
    
    batch_size = args.batch if args.batch is not None else \
                 INIT_BATCH_SIZE if INIT_BATCH_SIZE is not None else \
                 default_batch_size

    # --- 설정 확인 및 유효성 검사 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "="*50); print("자동 라벨링(Auto-Labeling)을 시작합니다."); print("="*50)
    print(f"  - 대상 데이터셋: {dataset_dir}")
    print(f"  - 선생님 모델: {teacher_model_path}")
    print(f"  - 신뢰도 임계값(conf): {conf_threshold}")
    print(f"  - 배치 사이즈(batch): {batch_size}")
    print(f"  - 실행 장치: {device.upper()}")
    print("="*50)

    images_dir = os.path.join(dataset_dir, 'images')
    if not os.path.isdir(images_dir):
        print(f"오류: 대상 데이터셋에 'images' 폴더가 없습니다. 경로: {images_dir}")
        return
    if not os.path.exists(teacher_model_path):
        print(f"오류: 선생님 모델 파일을 찾을 수 없습니다. 경로: {teacher_model_path}")
        return

    # 2. 모델 로드
    try:
        model = YOLO(teacher_model_path)
        model.to(device)
    except Exception as e:
        print(f"오류: 모델 로딩 중 오류 발생: {e}")
        return

    # 3. 입출력 경로 설정 및 이미지 파일 목록 가져오기
    labels_dir = os.path.join(dataset_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    image_formats = config['image_format'].split(',')
    image_paths = []
    for fmt in image_formats:
        image_paths.extend(glob.glob(os.path.join(images_dir, f'*.{fmt}')))
    
    if not image_paths:
        print(f"오류: '{images_dir}' 경로에 라벨링할 이미지가 없습니다.")
        return
        
    print(f"총 {len(image_paths)}개의 이미지를 대상으로 자동 라벨링을 진행합니다.")

    # 4. 배치 단위로 추론 및 라벨 파일 저장
    for i in tqdm(range(0, len(image_paths), batch_size), desc="자동 라벨링 진행 중"):
        batch_paths = image_paths[i:i + batch_size]
        
        try:
            results = model(batch_paths, conf=conf_threshold, verbose=False)
            
            for res in results:
                label_name = os.path.splitext(os.path.basename(res.path))[0] + '.txt'
                label_path = os.path.join(labels_dir, label_name)
                
                # YOLO 형식 (class_id x_center y_center width height)으로 변환하여 저장
                with open(label_path, 'w') as f:
                    for box in res.boxes:
                        class_id = int(box.cls)
                        x, y, w, h = box.xywhn[0]
                        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            print(f"\n오류: 추론 중 오류가 발생했습니다. (이미지: {batch_paths[0]}...)\n - {e}")
            continue

    print("\n자동 라벨링 작업이 완료되었습니다!")
    print(f"'{labels_dir}' 폴더에 라벨 파일들이 저장되었습니다.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("오류: _config.yaml 파일을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인하세요.")
        exit()

    parser = argparse.ArgumentParser(description="선생님 모델을 사용하여 이미지 데이터셋을 자동으로 라벨링합니다.")
    parser.add_argument('--dataset', type=str, default=None, help="자동 라벨링을 적용할 데이터셋의 상대 경로.")
    parser.add_argument('--weights', type=str, default=None, help="선생님 모델 가중치(.pt) 파일의 상대 경로.")
    parser.add_argument('--conf', type=float, default=None, help="객체 탐지 신뢰도 임계값.")
    parser.add_argument('--batch', type=int, default=None, help="추론 시 배치 사이즈.")
    args = parser.parse_args()

    auto_label_dataset(config, args)