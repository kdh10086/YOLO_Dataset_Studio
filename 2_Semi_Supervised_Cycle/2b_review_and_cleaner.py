import cv2
import os
import glob
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
INIT_DATASET_DIR = None      # 예시: 'datasets/track_dataset_0811'
INIT_MODE = 'visualize'      # 'visualize' 또는 'remove_noise'
INIT_CLEANED_DIR = None      # remove_noise 모드에서 사용할 출력 폴더. 예: 'datasets/cleaned_data'
# ==============================================================================

class Visualizer:
    """
    라벨링된 데이터셋을 시각화하고 검토/정제하는 클래스.
    """
    def __init__(self, dataset_dir, mode, cleaned_dir, config):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.cleaned_dir = cleaned_dir
        self.config = config
        
        self.classes = config['classes']
        self.colors = {cid: ((cid*40+50)%256, (cid*80+100)%256, (cid*120+150)%256) for cid in self.classes.keys()}
        
        self.is_paused = True
        self.img_index = 0
        self.review_files = set()
        self.noise_files = set()
        
        self.window_name = f"Label Visualizer - {self.mode.upper()} MODE"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self.image_paths, self.is_split = self._load_image_paths()

    def _load_image_paths(self):
        """데이터셋 구조(분할/통합)를 자동으로 감지하고 모든 이미지 경로를 로드합니다."""
        image_formats = self.config['image_format'].split(',')
        paths = []
        
        # 분할 구조인지 확인 (images/train 폴더 존재 여부 기준)
        is_split = os.path.isdir(os.path.join(self.dataset_dir, 'images', 'train'))
        
        if is_split:
            print("  - 데이터셋 구조: 분할됨 (train/val 폴더에서 이미지를 불러옵니다)")
            for sub_dir in ['train', 'val']:
                for fmt in image_formats:
                    paths.extend(glob.glob(os.path.join(self.dataset_dir, 'images', sub_dir, f'*.{fmt}')))
        else:
            print("  - 데이터셋 구조: 통합됨 (images 폴더에서 이미지를 바로 불러옵니다)")
            images_dir = os.path.join(self.dataset_dir, 'images')
            for fmt in image_formats:
                paths.extend(glob.glob(os.path.join(images_dir, f'*.{fmt}')))
        
        return sorted(paths), is_split

    def _mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 이벤트를 처리합니다."""
        if event == cv2.EVENT_LBUTTONDOWN:
            image_name = os.path.basename(self.image_paths[self.img_index])
            
            if self.mode == 'visualize':
                target_set = self.review_files
                msg = "검토 목록"
            else: # remove_noise
                target_set = self.noise_files
                msg = "제거 대상"

            if image_name not in target_set:
                target_set.add(image_name)
                print(f"  -> '{image_name}' {msg}에 추가됨 (총 {len(target_set)}개)")
            else:
                target_set.remove(image_name)
                print(f"  -> '{image_name}' {msg}에서 제거됨 (총 {len(target_set)}개)")

    def _create_cleaned_dataset(self):
        """노이즈로 지정된 프레임을 제외하고 새 데이터셋 폴더에 저장합니다."""
        print(f"\n[노이즈 제거] '{self.cleaned_dir}' 폴더에 정리된 데이터셋을 저장합니다.")

        if os.path.exists(self.cleaned_dir):
            print(f"  - 경고: 기존 '{self.cleaned_dir}' 폴더를 삭제하고 새로 생성합니다.")
            shutil.rmtree(self.cleaned_dir)
        
        cleaned_images_dir = os.path.join(self.cleaned_dir, 'images')
        cleaned_labels_dir = os.path.join(self.cleaned_dir, 'labels')
        os.makedirs(cleaned_images_dir, exist_ok=True)
        os.makedirs(cleaned_labels_dir, exist_ok=True)
        
        copied_count = 0
        for image_path in self.image_paths:
            image_filename = os.path.basename(image_path)
            
            if image_filename not in self.noise_files:
                label_filename = os.path.splitext(image_filename)[0] + '.txt'
                
                # 라벨 파일 경로 추적
                if self.is_split:
                    sub_dir = os.path.basename(os.path.dirname(image_path))
                    original_label_path = os.path.join(self.dataset_dir, 'labels', sub_dir, label_filename)
                else:
                    original_label_path = os.path.join(self.dataset_dir, 'labels', label_filename)
                
                # 새 경로에 이미지와 라벨 복사 (라벨은 통합된 구조로 저장)
                shutil.copy2(image_path, os.path.join(cleaned_images_dir, image_filename))
                if os.path.exists(original_label_path):
                    shutil.copy2(original_label_path, os.path.join(cleaned_labels_dir, label_filename))
                copied_count += 1

        print(f"  - 총 {len(self.image_paths)}개 중 {len(self.noise_files)}개의 노이즈 프레임을 제외하고,")
        print(f"  - {copied_count}개의 파일을 성공적으로 복사했습니다.")

    def run(self):
        """시각화 메인 루프를 실행합니다."""
        if not self.image_paths:
            print(f"오류: '{self.dataset_dir}' 경로에서 이미지를 찾을 수 없습니다.")
            return

        print("\n조작법: [Space]: 자동재생/일시정지 | [d]: 다음 | [a]: 이전 | [마우스클릭]: 선택/해제 | [q]: 종료")
        
        while 0 <= self.img_index < len(self.image_paths):
            image_path = self.image_paths[self.img_index]
            image_name = os.path.basename(image_path)
            
            # 라벨 파일 경로 추적
            label_filename = os.path.splitext(image_name)[0] + '.txt'
            if self.is_split:
                sub_dir = os.path.basename(os.path.dirname(image_path))
                label_path = os.path.join(self.dataset_dir, 'labels', sub_dir, label_filename)
            else:
                label_path = os.path.join(self.dataset_dir, 'labels', label_filename)

            img = cv2.imread(image_path)
            if img is None: self.img_index += 1; continue
            h, w = img.shape[:2]

            # 바운딩 박스 그리기
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        class_id, x_c, y_c, bw, bh = int(parts[0]), *map(float, parts[1:5])
                        x1, y1 = int((x_c - bw / 2) * w), int((y_c - bh / 2) * h)
                        x2, y2 = int((x_c + bw / 2) * w), int((y_c + bh / 2) * h)
                        color = self.colors.get(class_id, (255, 255, 255))
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label = f'{class_id}: {self.classes.get(class_id, "Unknown")}'
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 상태 텍스트 그리기
            if self.mode == 'visualize' and image_name in self.review_files:
                cv2.putText(img, "REVIEW", (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3)
            elif self.mode == 'remove_noise' and image_name in self.noise_files:
                cv2.putText(img, "TO BE REMOVED", (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (50, 50, 200), 3)
            
            cv2.imshow(self.window_name, img)
            
            key = cv2.waitKey(100 if not self.is_paused else -1) & 0xFF
            
            if key == ord('q'): break
            elif key == ord(' '): self.is_paused = not self.is_paused
            
            if self.is_paused:
                if key == ord('d'): self.img_index = min(self.img_index + 1, len(self.image_paths) - 1)
                elif key == ord('a'): self.img_index = max(self.img_index - 1, 0)
            else:
                self.img_index += 1
                
        # 종료 시 후처리
        if self.mode == 'visualize' and self.review_files:
            review_list_path = os.path.join(self.dataset_dir, 'review_list.txt')
            with open(review_list_path, 'w') as f:
                f.write('\n'.join(sorted(list(self.review_files))))
            print(f"\n'{review_list_path}' 파일에 검토 목록 저장 완료.")
        
        elif self.mode == 'remove_noise' and self.noise_files:
            self._create_cleaned_dataset()

        cv2.destroyAllWindows()
        print("\n시각화 도구를 종료합니다.")

def main(config, args):
    """설정값을 결정하고 Visualizer를 실행합니다."""
    project_root = os.path.dirname(os.path.abspath(__file__))

    dataset_dir_relative = args.dataset if args.dataset is not None else \
                           INIT_DATASET_DIR if INIT_DATASET_DIR is not None else \
                           config['datasets']['raw'] # visualize는 보통 대용량 데이터셋에 사용
    dataset_dir = os.path.join(project_root, dataset_dir_relative)
    
    mode = args.mode if args.mode is not None else INIT_MODE
    
    cleaned_dir_relative = args.output_dir if args.output_dir is not None else \
                           INIT_CLEANED_DIR if INIT_CLEANED_DIR is not None else \
                           config['datasets']['denoised']
    cleaned_dir = os.path.join(project_root, cleaned_dir_relative)

    print("\n" + "="*50)
    print("라벨 시각화 및 검토 도구를 시작합니다.")
    print("="*50)
    print(f"  - 대상 데이터셋: {dataset_dir}")
    print(f"  - 실행 모드: {mode}")
    if mode == 'remove_noise':
        print(f"  - 정제 데이터셋 출력 경로: {cleaned_dir}")
    print("="*50)

    visualizer = Visualizer(dataset_dir, mode, cleaned_dir, config)
    visualizer.run()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("오류: _config.yaml 파일을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인하세요.")
        exit()

    parser = argparse.ArgumentParser(description="라벨링된 데이터셋을 시각화하거나 노이즈를 제거합니다.")
    parser.add_argument('--dataset', type=str, default=None, help="작업할 데이터셋의 상대 경로.")
    parser.add_argument('--mode', type=str, default=None, choices=['visualize', 'remove_noise'], help="실행 모드 선택.")
    parser.add_argument('--output_dir', type=str, default=None, help="'remove_noise' 모드에서 정제된 데이터셋을 저장할 경로.")
    args = parser.parse_args()
    
    main(config, args)