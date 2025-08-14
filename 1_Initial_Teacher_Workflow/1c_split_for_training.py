import os
import glob
import random
import shutil
import yaml
import argparse

# ==============================================================================
# 초기 실행 설정 (Initiation Settings)
#
# - Code Runner 또는 IDE에서 '직접 실행' 시 이 부분을 수정하여 사용하세요.
# - 터미널에서 인자를 직접 지정하면 이 설정은 무시됩니다.
# - 값을 None으로 두면 _config.yaml의 기본 설정을 따릅니다.
# ==============================================================================
INIT_DATASET_DIR = None   # 예시: 'datasets/my_split_data'
# ==============================================================================


def split_and_organize_files(dataset_dir, train_ratio, image_formats):
    """
    데이터셋 폴더 내의 이미지와 라벨 파일들을 train/val로 분할합니다.
    이미 분할된 경우, 파일을 초기화하고 다시 분할합니다.
    """
    print("\n[STEP 1] 데이터셋 분할을 시작합니다...")

    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    # 이미 분할된 폴더(train/val)가 있다면, 모든 파일을 다시 상위 폴더로 이동시켜 초기화
    for sub_dir_name in ['train', 'val']:
        for content_type_dir in [images_dir, labels_dir]:
            sub_dir_path = os.path.join(content_type_dir, sub_dir_name)
            if os.path.isdir(sub_dir_path):
                print(f"  - 기존 폴더 '{os.path.basename(content_type_dir)}/{sub_dir_name}'의 파일들을 초기화합니다.")
                for filename in os.listdir(sub_dir_path):
                    shutil.move(os.path.join(sub_dir_path, filename), content_type_dir)
                os.rmdir(sub_dir_path)

    # 분할될 폴더 경로 정의 및 생성
    train_img_dir = os.path.join(images_dir, 'train')
    val_img_dir = os.path.join(images_dir, 'val')
    train_lbl_dir = os.path.join(labels_dir, 'train')
    val_lbl_dir = os.path.join(labels_dir, 'val')
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # 이미지 파일 목록을 가져와서 이름순으로 정렬 후, 무작위로 섞음
    image_paths = []
    for fmt in image_formats:
        image_paths.extend(glob.glob(os.path.join(images_dir, f'*.{fmt}')))
    
    if not image_paths:
        print(f"오류: '{images_dir}'에 분할할 이미지가 없습니다.")
        return False
        
    random.shuffle(image_paths)
    
    # 학습용과 검증용으로 분할
    split_point = int(len(image_paths) * train_ratio)
    train_files = image_paths[:split_point]
    val_files = image_paths[split_point:]
    
    labeled_train_count = 0
    labeled_val_count = 0

    # 파일 이동
    for img_path in train_files:
        image_name = os.path.basename(img_path)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        if os.path.exists(label_path):
            shutil.move(img_path, os.path.join(train_img_dir, image_name))
            shutil.move(label_path, os.path.join(train_lbl_dir, label_name))
            labeled_train_count += 1
        else:
            print(f"  - 경고: '{image_name}'에 해당하는 라벨 파일이 없어 건너뜁니다.")
    
    for img_path in val_files:
        image_name = os.path.basename(img_path)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        if os.path.exists(label_path):
            shutil.move(img_path, os.path.join(val_img_dir, image_name))
            shutil.move(label_path, os.path.join(val_lbl_dir, label_name))
            labeled_val_count += 1
        else:
            print(f"  - 경고: '{image_name}'에 해당하는 라벨 파일이 없어 건너뜁니다.")

    print("데이터셋 분할 완료!")
    print(f"  - 학습용: {labeled_train_count}개 | 검증용: {labeled_val_count}개")
    return True

def generate_data_yaml(dataset_dir, classes_dict):
    """
    YOLO 학습에 필요한 data.yaml 파일을 생성합니다.
    """
    print("\n[STEP 2] data.yaml 파일 생성을 시작합니다...")

    # 어떤 환경에서든 학습이 가능하도록 데이터셋의 절대 경로를 사용
    dataset_abs_path = os.path.abspath(dataset_dir)
    class_names = [name for i, name in sorted(classes_dict.items())]

    yaml_content = {
        'path': dataset_abs_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': class_names
    }

    yaml_file_path = os.path.join(dataset_dir, 'data.yaml')

    try:
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
        print(f"data.yaml 파일이 성공적으로 생성되었습니다: {yaml_file_path}")
        print("--- YAML 내용 ---")
        print(yaml.dump(yaml_content, allow_unicode=True, sort_keys=False).strip())
        print("-------------------")
    except Exception as e:
        print(f"오류: data.yaml 파일 생성 중 오류 발생: {e}")

def main(config, args):
    """
    데이터셋 분할 및 data.yaml 생성 작업을 총괄합니다.
    """
    # 1. 설정값 결정 (3단계 우선순위 적용)
    project_root = os.path.dirname(os.path.abspath(__file__))

    dataset_dir_relative = args.dataset if args.dataset is not None else \
                           INIT_DATASET_DIR if INIT_DATASET_DIR is not None else \
                           config['datasets']['sample']
    dataset_dir = os.path.join(project_root, dataset_dir_relative)

    train_ratio = config['train_ratio']
    image_formats = config['image_format'].split(',')
    classes_dict = config['classes']

    # --- 설정 확인 및 출력 ---
    print("\n" + "="*50)
    print("데이터셋 분할 및 YAML 생성을 시작합니다.")
    print("="*50)
    print(f"  - 대상 데이터셋: {dataset_dir}")
    print(f"  - 분할 비율 (Train): {train_ratio}")
    print("="*50)

    # 2. 작업 실행
    if not os.path.isdir(os.path.join(dataset_dir, 'images')):
        print(f"오류: 대상 폴더에 'images' 디렉토리가 없습니다. 경로를 확인하세요: {dataset_dir}")
        return

    if split_and_organize_files(dataset_dir, train_ratio, image_formats):
        generate_data_yaml(dataset_dir, classes_dict)
        print("\n모든 작업이 성공적으로 완료되었습니다.")
    else:
        print("\n분할 작업에 실패하여 YAML 파일을 생성하지 않았습니다.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("오류: _config.yaml 파일을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인하세요.")
        exit()

    parser = argparse.ArgumentParser(description="데이터셋을 train/val로 분할하고 data.yaml을 생성합니다.")
    parser.add_argument('--dataset', type=str, default=None, help="분할할 데이터셋의 상대 경로. (예: 'datasets/sample_dataset')")
    args = parser.parse_args()

    main(config, args)