import cv2
import os
import glob
import yaml
import argparse

# ==============================================================================
# 초기 실행 설정 (Initiation Settings)
#
# - Code Runner 또는 IDE에서 '직접 실행' 시 이 부분을 수정하여 사용하세요.
# - 터미널에서 인자를 직접 지정하면 이 설정은 무시됩니다.
# - 값을 None으로 두면 _config.yaml의 기본 설정을 따릅니다.
# ==============================================================================
INIT_DATASET_DIR = None   # 예시: 'datasets/my_labeling_data'
INIT_IMAGE_NAME = None    # 예시: '000123.png'
# ==============================================================================


def main(config, args):
    """
    분할되지 않은 데이터셋의 라벨을 시각화합니다.
    """
    # 1. 설정값 결정 (3단계 우선순위 적용)
    project_root = os.path.dirname(os.path.abspath(__file__))

    dataset_dir_relative = args.dataset if args.dataset is not None else \
                           INIT_DATASET_DIR if INIT_DATASET_DIR is not None else \
                           config['datasets']['sample']
    dataset_dir = os.path.join(project_root, dataset_dir_relative)

    start_image_name = args.start_image if args.start_image is not None else \
                       INIT_IMAGE_NAME

    # --- 설정 확인 및 출력 ---
    print("\n" + "="*50)
    print("데이터셋 시각화 도구를 시작합니다.")
    print("="*50)
    print(f"  - 대상 데이터셋: {dataset_dir}")
    if start_image_name:
        print(f"  - 시작 이미지: {start_image_name}")
    print("="*50)

    # 2. 데이터셋 경로 및 파일 목록 설정
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    classes_dict = config['classes']
    colors = {cid: ((cid * 55 + 50) % 256, (cid * 95 + 100) % 256, (cid * 135 + 150) % 256) for cid in classes_dict.keys()}
    
    image_formats = config['image_format'].split(',')
    image_paths = []
    for ext in image_formats:
        image_paths.extend(glob.glob(os.path.join(images_dir, f'*.{ext}')))
    
    if not image_paths:
        print(f"오류: '{images_dir}'에서 이미지를 찾을 수 없습니다.")
        return
        
    image_paths.sort()
    total_images = len(image_paths)

    # 3. 시작 인덱스 결정
    img_index = 0
    if start_image_name:
        try:
            # os.path.basename을 사용하여 순수 파일 이름으로만 리스트를 만듦
            image_filenames = [os.path.basename(p) for p in image_paths]
            img_index = image_filenames.index(start_image_name)
        except ValueError:
            print(f"경고: 시작 이미지 '{start_image_name}'을(를) 찾을 수 없어 처음부터 시작합니다.")

    # 4. 시각화 메인 루프
    print("\n조작법: [d] 또는 [Space]: 다음 | [a]: 이전 | [q]: 종료")
    window_name = "Label Visualizer"
    
    while 0 <= img_index < total_images:
        image_path = image_paths[img_index]
        image_name = os.path.basename(image_path)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')

        img = cv2.imread(image_path)
        if img is None:
            print(f"경고: '{image_name}' 이미지를 불러올 수 없습니다. 건너뜁니다.")
            img_index += 1
            continue
            
        h, w = img.shape[:2]

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    class_id, x_c, y_c, bw, bh = int(parts[0]), *map(float, parts[1:])
                    
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)
                    
                    color = colors.get(class_id, (255, 255, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    label = f'{class_id}: {classes_dict.get(class_id, "Unknown")}'
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # 라벨이 없는 경우, 화면에 경고 문구 표시
            cv2.putText(img, "LABEL NOT FOUND", (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)
            print(f"경고: '{image_name}'에 해당하는 라벨 파일이 없습니다.")

        title = f"[{img_index + 1}/{total_images}] {image_name}"
        cv2.setWindowTitle(window_name, title)
        cv2.imshow(window_name, img)
        
        # 키 입력 대기
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d') or key == ord(' '): # d 또는 스페이스바
            img_index += 1
        elif key == ord('a'):
            img_index -= 1
    
    print("\n시각화 도구를 종료합니다.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("오류: _config.yaml 파일을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인하세요.")
        exit()

    parser = argparse.ArgumentParser(description="분할되지 않은 데이터셋의 라벨을 시각화합니다.")
    parser.add_argument('--dataset', type=str, default=None, help="시각화할 데이터셋의 상대 경로. (예: 'datasets/sample_dataset')")
    parser.add_argument('--start_image', type=str, default=None, help="시각화를 시작할 이미지 파일 이름. (예: '000123.png')")
    args = parser.parse_args()
    
    main(config, args)