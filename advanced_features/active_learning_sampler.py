import os
import glob
import shutil
import numpy as np
import torch
import yaml
import argparse
from PIL import Image
from sklearn.cluster import KMeans
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ==============================================================================
# 초기 실행 설정 (Initiation Settings)
# ==============================================================================
INIT_SOURCE_DIR = None      # 예시: 'datasets/unlabeled_pool'
INIT_WEIGHTS_PATH = None    # 예시: 'runs/train/teacher_v1/weights/best.pt'
INIT_WORK_DIR = None        # 예시: 'my_al_workspace'
INIT_SELECTION_SIZE = None  # 예시: 200
INIT_MIN_CONF = None        # 예시: 0.3
INIT_MAX_CONF = None        # 예시: 0.7
# ==============================================================================


class ActiveLearningSampler:
    """
    불확실성 및 다양성 샘플링을 기반으로 액티브 러닝을 수행하는 클래스.
    """
    def __init__(self, config, args):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._setup_config(config, args)
        self._setup_directories()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"실행 장치: {self.device.upper()}")

        self.yolo_model = YOLO(self.weights_path)
        self.feature_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    def _setup_config(self, config, args):
        """3단계 우선순위에 따라 설정을 초기화합니다."""
        defaults = config.get('active_learning_defaults', {})
        
        self.source_dir = os.path.join(self.project_root, args.source or INIT_SOURCE_DIR or defaults.get('source_dir'))
        self.weights_path = os.path.join(self.project_root, args.weights or INIT_WEIGHTS_PATH or defaults.get('weights_path'))
        self.work_dir = os.path.join(self.project_root, args.workdir or INIT_WORK_DIR or defaults.get('work_dir'))
        
        self.selection_size = args.size if args.size is not None else INIT_SELECTION_SIZE if INIT_SELECTION_SIZE is not None else defaults.get('selection_size', 100)
        self.min_conf = args.min_conf if args.min_conf is not None else INIT_MIN_CONF if INIT_MIN_CONF is not None else defaults.get('min_conf', 0.4)
        self.max_conf = args.max_conf if args.max_conf is not None else INIT_MAX_CONF if INIT_MAX_CONF is not None else defaults.get('max_conf', 0.8)

    def _setup_directories(self):
        """작업에 필요한 디렉토리들을 설정하고 생성합니다."""
        self.predict_dir = os.path.join(self.work_dir, 'predictions')
        self.predict_labels_dir = os.path.join(self.predict_dir, 'labels')
        self.feature_dir = os.path.join(self.work_dir, 'features')
        self.selection_dir = os.path.join(self.work_dir, 'selected_for_labeling')
        
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.predict_labels_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.selection_dir, exist_ok=True)

    def _run_predictions(self):
        """1단계: YOLO 모델로 전체 소스 데이터에 대한 예측을 수행합니다."""
        print("\n--- [단계 1/5] 모델 예측 수행 ---")
        if os.listdir(self.predict_labels_dir):
            print("이미 예측 결과가 존재하므로 이 단계를 건너뜁니다.")
            return
        
        print(f"'{self.source_dir}'의 모든 이미지에 대해 예측을 시작합니다...")
        self.yolo_model.predict(
            source=self.source_dir, save_txt=True, save_conf=True,
            project=os.path.dirname(self.predict_dir), name=os.path.basename(self.predict_dir),
            exist_ok=True, verbose=False, stream=True
        )
        print("예측 완료. 라벨 파일이 저장되었습니다.")

    def _extract_features(self):
        """2단계: CLIP 모델로 모든 이미지의 특징 벡터를 추출합니다."""
        print("\n--- [단계 2/5] 특징 벡터 추출 ---")
        image_paths = glob.glob(os.path.join(self.source_dir, '**', '*.*'), recursive=True)
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

        with torch.no_grad():
            for img_path in tqdm(image_paths, desc="특징 추출 중"):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                feature_path = os.path.join(self.feature_dir, f"{base_name}.npy")
                if os.path.exists(feature_path): continue

                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
                    embedding = self.feature_model.get_image_features(inputs)
                    np.save(feature_path, embedding.cpu().numpy())
                except Exception as e:
                    print(f"경고: {img_path} 처리 중 오류 발생 - {e}")

        print("특징 벡터 추출 완료.")

    def _select_uncertain_candidates(self):
        """3단계: 불확실성 샘플링으로 1차 후보군을 선별합니다."""
        print("\n--- [단계 3/5] 불확실성 기반 후보군 선별 ---")
        txt_files = glob.glob(os.path.join(self.predict_labels_dir, '*.txt'))
        candidate_basenames = []
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                    if not lines: continue
                    confidences = [float(line.strip().split()[5]) for line in lines if len(line.strip().split()) > 5]
                    if not confidences: continue
                    
                    avg_conf = np.mean(confidences)
                    if self.min_conf <= avg_conf <= self.max_conf:
                        candidate_basenames.append(os.path.splitext(os.path.basename(txt_file))[0])
            except Exception: continue
        
        print(f"총 {len(candidate_basenames)}개의 불확실한 후보를 1차 선별했습니다.")
        return candidate_basenames

    def _select_diverse_subset(self, candidate_basenames):
        """4단계: K-Means 클러스터링으로 다양한 최종 서브셋을 선별합니다."""
        print("\n--- [단계 4/5] 다양성 기반 최종 샘플 선별 ---")
        features, valid_basenames = [], []
        for name in candidate_basenames:
            feature_path = os.path.join(self.feature_dir, f"{name}.npy")
            if os.path.exists(feature_path):
                features.append(np.load(feature_path).flatten())
                valid_basenames.append(name)
        
        if not features or len(features) <= self.selection_size:
            print("후보군 수가 최종 목표보다 적거나 없어, 모든 후보를 최종 선택합니다.")
            return valid_basenames

        print(f"{len(features)}개의 후보군을 {self.selection_size}개의 클러스터로 그룹화합니다...")
        features = np.array(features)
        kmeans = KMeans(n_clusters=self.selection_size, random_state=42, n_init='auto').fit(features)
        
        final_selection_indices = []
        for i in range(self.selection_size):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) == 0: continue
            
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(features[cluster_indices] - centroid, axis=1)
            closest_point_index_in_cluster = np.argmin(distances)
            original_index = cluster_indices[closest_point_index_in_cluster]
            final_selection_indices.append(original_index)

        final_selection_basenames = [valid_basenames[i] for i in final_selection_indices]
        print(f"다양성이 높은 최종 이미지 {len(final_selection_basenames)}개를 선별했습니다.")
        return final_selection_basenames

    def _copy_selected_files(self, basenames):
        """5단계: 최종 선별된 이미지들을 지정된 폴더로 복사합니다."""
        print(f"\n--- [단계 5/5] 최종 선별된 파일 복사 ---")
        image_paths = glob.glob(os.path.join(self.source_dir, '**', '*.*'), recursive=True)
        path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))}
        
        copied_count = 0
        for name in tqdm(basenames, desc="선별된 파일 복사 중"):
            if name in path_map:
                shutil.copy2(path_map[name], self.selection_dir)
                copied_count += 1
        
        print(f"성공적으로 {copied_count}개의 파일을 '{self.selection_dir}' 폴더로 복사했습니다.")

    def run(self):
        """액티브 러닝 파이프라인 전체를 실행합니다."""
        self._run_predictions()
        self._extract_features()
        uncertain_candidates = self._select_uncertain_candidates()
        if uncertain_candidates:
            final_selections = self._select_diverse_subset(uncertain_candidates)
            self._copy_selected_files(final_selections)
        print("\n모든 작업이 완료되었습니다.")

def main(config, args):
    """설정값을 결정하고 ActiveLearningSampler를 실행합니다."""
    sampler = ActiveLearningSampler(config, args)
    sampler.run()

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(project_root, '_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("오류: _config.yaml 파일을 찾을 수 없습니다."); exit()

    parser = argparse.ArgumentParser(description="액티브 러닝을 통해 라벨링할 이미지를 지능적으로 샘플링합니다.")
    parser.add_argument('--source', type=str, default=None, help="이미지 풀이 있는 소스 데이터셋의 상대 경로.")
    parser.add_argument('--weights', type=str, default=None, help="예측에 사용할 선생님 모델(.pt)의 상대 경로.")
    parser.add_argument('--workdir', type=str, default=None, help="중간 결과물이 저장될 작업 공간의 상대 경로.")
    parser.add_argument('--size', type=int, default=None, help="최종적으로 선택할 이미지 개수.")
    parser.add_argument('--min_conf', type=float, default=None, help="불확실성 샘플링의 최소 신뢰도.")
    parser.add_argument('--max_conf', type=float, default=None, help="불확실성 샘플링의 최대 신뢰도.")
    args = parser.parse_args()
    
    main(config, args)