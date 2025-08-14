import os
import glob
import shutil
import random
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
INIT_SOURCE_DIR = None      # 예시: 'datasets/track_dataset_0811'
INIT_OUTPUT_DIR = None      # 예시: 'datasets/my_sampled_data'
INIT_SAMPLE_RATIO = None    # 예시: 0.05 (5%)
# ==============================================================================


def main(config, args):
    """
    대규모 데이터셋에서 지정된 비율만큼의 데이터를 무작위로 샘플링하여
    새로운 평면 구조의 데이터셋을 생성합니다.
    """
    # 1. 설정값 결정 (3단계 우선순위 적용)
    project_root = os.path.dirname(os.path.abspath(__file__))
    defaults = config.get('sampling_defaults', {})

    source_dir_relative = args.source if args.source is not None else \
                          INIT_SOURCE_DIR if INIT_SOURCE_DIR is not None else \
                          defaults.get('source')
    source_dir = os.path.join(project_root, source_dir_relative)

    output_dir_relative = args.output if args.output is not None else \
                          INIT_OUTPUT_DIR if INIT_OUTPUT_DIR is not None else \
                          defaults.get('output')
    output_dir = os.path.join(project_root, output_dir_relative)
    
    sample_ratio = args.ratio if args.ratio is not None else \
                   INIT_SAMPLE_RATIO if INIT_SAMPLE_RATIO is not None else \
                   defaults.get('ratio', 0.1)

    # --- 설정 확인 및 유효성 검사 ---
    print("\n" + "="*50); print("데이터셋 무작위 샘플링을 시작합니다."); print("="*50)
    print(f"  - 원본 데이터셋: {source_dir}")
    print(f"  - 샘플 저장 경로: {output_dir}")
    print(f"  - 샘플링 비율: {sample_ratio:.2%}")
    print("="*50)
    
    if not os.path.isdir(source_dir):
        print(f"오류: 원본 데이터셋 경로를 찾을 수 없습니다: {source_dir}"); return
    if not (0.0 < sample_ratio <= 1.0):
        print(f"오류: 샘플링 비율은 0과 1 사이의 값이어야 합니다: {sample_ratio}"); return

    # 2. 모든 이미지 파일 검색 및 라벨 파일과 쌍으로 만들기
    print("원본 데이터셋에서 모든 이미지 파일을 검색 중입니다...")
    image_formats = config['image_format'].split(',')
    all_image_paths = []
    # 원본 데이터셋의 하위 폴더까지 모두 검색 (재귀적)
    for fmt in image_formats:
        all_image_paths.extend(glob.glob(os.path.join(source_dir, '**', f'*.{fmt}'), recursive=True))

    file_pairs = []
    for img_path in all_image_paths:
        # OS에 맞는 경로 구분자로 'images'를 'labels'로 변경하여 라벨 경로 추정
        label_path = os.path.splitext(img_path.replace(f'{os.path.sep}images{os.path.sep}', f'{os.path.sep}labels{os.path.sep}', 1))[0] + '.txt'
        if os.path.exists(label_path):
            file_pairs.append({'image': img_path, 'label': label_path})
    
    if not file_pairs:
        print("오류: 원본 데이터셋에서 라벨과 쌍을 이루는 이미지를 찾을 수 없습니다."); return
    print(f"총 {len(file_pairs)}개의 이미지-라벨 쌍을 찾았습니다.")

    # 3. 랜덤 셔플 및 샘플링
    random.shuffle(file_pairs)
    num_to_sample = int(len(file_pairs) * sample_ratio)
    sampled_files = file_pairs[:num_to_sample]
    print(f"이 중 {num_to_sample}개를 샘플링합니다.")

    # 4. 파일 복사 (평면 구조로)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_images_dir)
    os.makedirs(output_labels_dir)

    print("\n샘플 파일 복사를 시작합니다...")
    for pair in tqdm(sampled_files, desc="파일 복사 중"):
        # 원본 파일명 그대로 복사
        shutil.copy2(pair['image'], os.path.join(output_images_dir, os.path.basename(pair['image'])))
        shutil.copy2(pair['label'], os.path.join(output_labels_dir, os.path.basename(pair['label'])))

    print("\n샘플링이 완료되었습니다!")
    print(f"  - 총 {len(sampled_files)}개의 파일이 '{output_dir}'에 저장되었습니다.")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("오류: _config.yaml 파일을 찾을 수 없습니다. 프로젝트 루트에 파일이 있는지 확인하세요."); exit()

    parser = argparse.ArgumentParser(description="대규모 데이터셋에서 일부를 무작위로 샘플링합니다.")
    parser.add_argument('--source', type=str, default=None, help="샘플링할 원본 데이터셋의 상대 경로.")
    parser.add_argument('--output', type=str, default=None, help="생성될 샘플 데이터셋의 상대 경로.")
    parser.add_argument('--ratio', type=float, default=None, help="샘플링 비율 (0.0 ~ 1.0).")
    args = parser.parse_args()
    
    main(config, args)