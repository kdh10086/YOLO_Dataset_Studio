import os
import glob
import shutil
import yaml
import argparse
from tqdm import tqdm

# ==============================================================================
# 초기 실행 설정 (Initiation Settings)
#
# - Code Runner 또는 IDE에서 '직접 실행' 시 이 부분을 수정하여 사용하세요.
# - 터미널에서 인자를 직접 지정하면 이 설정은 무시됩니다.
# - 값을 None으로 두면 _config.yaml의 기본 설정을 따릅니다.
# ==============================================================================
INIT_INPUT_DIRS = None   # 예시: ['datasets/data1', 'datasets/data2']
INIT_OUTPUT_DIR = None   # 예시: 'datasets/my_merged_data'
# ==============================================================================

def find_all_image_paths(dataset_dir):
    """
    데이터셋 디렉토리 구조(통합/분할)를 자동으로 감지하고 모든 이미지 파일 경로를 반환합니다.
    """
    image_paths = []
    image_formats = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))

    is_split = os.path.isdir(os.path.join(dataset_dir, 'images', 'train'))
    
    print(f"  - '{dataset_name}' 데이터셋 분석 중... (구조: {'분할됨' if is_split else '통합됨'})")
    
    base_images_dir = os.path.join(dataset_dir, 'images')
    dirs_to_scan = [os.path.join(base_images_dir, sub) for sub in ['train', 'val']] if is_split else [base_images_dir]

    for d in dirs_to_scan:
        for fmt in image_formats:
            image_paths.extend(glob.glob(os.path.join(d, fmt)))
            
    return sorted(image_paths)

def merge_and_rename(all_image_paths, output_dir):
    """
    주어진 모든 이미지와 라벨 파일들을 하나의 폴더로 병합하고,
    파일명을 0부터 순차적으로 새로 부여합니다.
    """
    if os.path.exists(output_dir):
        print(f"\n경고: 기존 출력 폴더 '{output_dir}'를 삭제하고 새로 생성합니다.")
        shutil.rmtree(output_dir)
    
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_images_dir)
    os.makedirs(output_labels_dir)

    print(f"\n총 {len(all_image_paths)}개의 파일을 병합 및 이름 변경합니다...")
    
    for i, img_path in enumerate(tqdm(all_image_paths, desc="파일 병합 중")):
        new_basename = f"{i:06d}"
        ext = os.path.splitext(img_path)[1]
        
        new_image_filename = f"{new_basename}{ext}"
        new_label_filename = f"{new_basename}.txt"
        
        label_path = os.path.splitext(img_path.replace('images', 'labels', 1))[0] + '.txt'

        shutil.copy2(img_path, os.path.join(output_images_dir, new_image_filename))
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(output_labels_dir, new_label_filename))

    print("\n병합 및 이름 변경 완료!")
    print(f"  - 총 {len(all_image_paths)}개의 파일이 '{output_dir}'에 저장되었습니다.")

def main(config, args):
    """스크립트의 메인 실행 로직."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. 설정값 결정 (3단계 우선순위 적용)
    input_dirs_relative = args.inputs if args.inputs is not None else \
                          INIT_INPUT_DIRS if INIT_INPUT_DIRS is not None else \
                          config.get('merge_defaults', {}).get('inputs', [])

    output_dir_relative = args.output if args.output is not None else \
                          INIT_OUTPUT_DIR if INIT_OUTPUT_DIR is not None else \
                          config.get('merge_defaults', {}).get('output')

    if not input_dirs_relative or not output_dir_relative:
        print("오류: 입력 또는 출력 데이터셋이 지정되지 않았습니다.")
        print("터미널 인자, 스크립트 상단 INIT 변수, 또는 _config.yaml 파일을 확인하세요.")
        return

    # --- 설정 확인 및 출력 ---
    print("\n" + "="*50); print("데이터셋 병합을 시작합니다."); print("="*50)
    print("입력 데이터셋 목록:")
    for path in input_dirs_relative:
        print(f"  - {path}")
    print(f"\n출력 데이터셋:\n  - {output_dir_relative}")
    print("="*50)

    # 2. 모든 이미지 경로 수집
    all_image_paths_to_merge = []
    for rel_path in input_dirs_relative:
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.isdir(abs_path):
            print(f"오류: 입력 데이터셋 경로를 찾을 수 없습니다 -> {abs_path}"); return
        
        image_paths = find_all_image_paths(abs_path)
        print(f"    -> '{rel_path}'에서 {len(image_paths)}개의 이미지 발견")
        all_image_paths_to_merge.extend(image_paths)

    output_abs_path = os.path.join(project_root, output_dir_relative)
    
    # 3. 병합 실행
    merge_and_rename(all_image_paths_to_merge, output_abs_path)
    
    print("\n모든 작업이 성공적으로 완료되었습니다.")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("오류: _config.yaml 파일을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인하세요.")
        exit()
        
    parser = argparse.ArgumentParser(description="여러 데이터셋을 하나의 새 데이터셋으로 병합합니다.")
    parser.add_argument('--inputs', nargs='+', default=None, help="병합할 데이터셋들의 상대 경로 목록.")
    parser.add_argument('--output', type=str, default=None, help="생성될 최종 데이터셋의 상대 경로.")
    args = parser.parse_args()
    
    main(config, args)