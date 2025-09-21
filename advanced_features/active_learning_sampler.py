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
# Initiation Settings
# ==============================================================================
# These global variables define default parameters for the active learning sampler.
# They can be overridden by command-line arguments or settings in models_config.yaml.

# Default path to the source directory containing the unlabeled image pool.
INIT_SOURCE_DIR = None

# Default path to the model weights (.pt file) used for making predictions.
INIT_WEIGHTS_PATH = None

# Default path to the workspace directory for saving intermediate and final results.
INIT_WORK_DIR = None

# Default number of images to select in the final diverse subset.
INIT_SELECTION_SIZE = None

# Default minimum average confidence for an image to be considered uncertain.
INIT_MIN_CONF = None

# Default maximum average confidence for an image to be considered uncertain.
INIT_MAX_CONF = None

# Default setting for overwriting existing prediction directories.
INIT_EXIST_OK = False
# ==============================================================================

class ActiveLearningSampler:
    """
    Performs active learning based on uncertainty and diversity sampling.
    """
    def __init__(self, args):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._setup_config(args)
        self._setup_directories()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Device: {self.device.upper()}")
        self.yolo_model = YOLO(self.weights_path)
        self.feature_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    def _setup_config(self, args):
        # Paths are now CLI arguments or prompted at runtime
        source_rel = args.source or input("Enter relative path to the source dataset with the image pool: ").strip()
        weights_rel = args.weights or input("Enter relative path to the teacher model weights (.pt): ").strip()
        workdir_rel = args.workdir or input("Enter relative path to the workspace directory: ").strip()

        self.source_dir = os.path.join(self.project_root, source_rel)
        self.weights_path = os.path.join(self.project_root, weights_rel)
        self.work_dir = os.path.join(self.project_root, workdir_rel)
        
        # Non-path parameters can still have defaults or be passed via CLI
        self.selection_size = args.size or 100
        self.min_conf = args.min_conf or 0.4
        self.max_conf = args.max_conf or 0.8
        self.exist_ok = args.exist_ok or False

        print("--- Active Learning Sampler Configuration ---")
        print(f"  - Source: {self.source_dir}")
        print(f"  - Weights: {self.weights_path}")
        print(f"  - Workspace: {self.work_dir}")
        print(f"  - Selection Size: {self.selection_size}")
        print(f"  - Confidence Range: [{self.min_conf}, {self.max_conf}]")
        print("-------------------------------------------")

    def _setup_directories(self):
        self.predict_dir = os.path.join(self.work_dir, 'predictions')
        self.predict_labels_dir = os.path.join(self.predict_dir, 'labels')
        self.feature_dir = os.path.join(self.work_dir, 'features')
        self.selection_dir = os.path.join(self.work_dir, 'selected_for_labeling')
        for d in [self.work_dir, self.predict_labels_dir, self.feature_dir, self.selection_dir]:
            os.makedirs(d, exist_ok=True)

    def _run_predictions(self):
        """Step 1: Run predictions on the entire source data with the YOLO model."""
        print("\n--- [Step 1/5] Running Model Predictions ---")
        if os.listdir(self.predict_labels_dir) and not self.exist_ok:
            print("Prediction results already exist and exist_ok is False, skipping this step.")
            return
        print(f"Starting predictions for all images in '{self.source_dir}'...")
        self.yolo_model.predict(source=self.source_dir, save_txt=True, save_conf=True,
                                project=os.path.dirname(self.predict_dir), name=os.path.basename(self.predict_dir),
                                exist_ok=self.exist_ok, verbose=False, stream=True)
        print("Predictions complete. Label files have been saved.")

    def _extract_features(self):
        """Step 2: Extract feature vectors from all images using the CLIP model."""
        print("\n--- [Step 2/5] Extracting Feature Vectors ---")
        img_paths = glob.glob(os.path.join(self.source_dir, '**', '*.*'), recursive=True)
        img_paths = [p for p in img_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        with torch.no_grad():
            for img_path in tqdm(img_paths, desc="Extracting features"):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                feature_path = os.path.join(self.feature_dir, f"{base_name}.npy")
                if os.path.exists(feature_path): continue
                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
                    embedding = self.feature_model.get_image_features(inputs)
                    np.save(feature_path, embedding.cpu().numpy())
                except Exception as e: print(f"Warning: Error processing {img_path} - {e}")
        print("Feature vector extraction complete.")

    def _select_uncertain_candidates(self):
        """Step 3: Select an initial pool of candidates via uncertainty sampling."""
        print("\n--- [Step 3/5] Selecting Candidates based on Uncertainty ---")
        candidate_basenames = []
        for txt_file in glob.glob(os.path.join(self.predict_labels_dir, '*.txt')):
            try:
                with open(txt_file, 'r') as f: lines = f.readlines()
                if not lines: continue
                confidences = [float(line.strip().split()[5]) for line in lines if len(line.strip().split()) > 5]
                if confidences and self.min_conf <= np.mean(confidences) <= self.max_conf:
                    candidate_basenames.append(os.path.splitext(os.path.basename(txt_file))[0])
            except Exception: continue
        print(f"Selected {len(candidate_basenames)} uncertain candidates in the first pass.")
        return candidate_basenames

    def _select_diverse_subset(self, candidate_basenames):
        """Step 4: Select the final diverse subset using K-Means clustering."""
        print("\n--- [Step 4/5] Selecting Final Samples based on Diversity ---")
        features, valid_basenames = [], []
        for name in candidate_basenames:
            feature_path = os.path.join(self.feature_dir, f"{name}.npy")
            if os.path.exists(feature_path):
                features.append(np.load(feature_path).flatten()); valid_basenames.append(name)
        if not features or len(features) <= self.selection_size:
            print("Number of candidates is less than or equal to the target size. Selecting all candidates.")
            return valid_basenames
        print(f"Clustering {len(features)} candidates into {self.selection_size} groups...")
        features = np.array(features)
        kmeans = KMeans(n_clusters=self.selection_size, random_state=42, n_init='auto').fit(features)
        final_indices = [cluster_indices[np.argmin(np.linalg.norm(features[cluster_indices] - kmeans.cluster_centers_[i], axis=1))]
                         for i in range(self.selection_size) if len(cluster_indices := np.where(kmeans.labels_ == i)[0]) > 0]
        final_basenames = [valid_basenames[i] for i in final_indices]
        print(f"Selected {len(final_basenames)} final images with high diversity.")
        return final_basenames

    def _copy_selected_files(self, basenames):
        """Step 5: Copy the final selected images to the designated folder."""
        print(f"\n--- [Step 5/5] Copying Final Selected Files ---")
        img_paths = glob.glob(os.path.join(self.source_dir, '**', '*.*'), recursive=True)
        path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))}
        
        copied_count = 0
        for name in tqdm(basenames, desc="Copying selected files"):
            if name in path_map:
                try:
                    shutil.copy2(path_map[name], self.selection_dir)
                    copied_count += 1
                except Exception as e:
                    print(f"Warning: Could not copy file for '{name}': {e}")

        print(f"Successfully copied {copied_count} files to the '{self.selection_dir}' folder.")

    def run(self):
        """Runs the entire active learning pipeline."""
        self._run_predictions(); self._extract_features()
        uncertain_candidates = self._select_uncertain_candidates()
        if uncertain_candidates:
            final_selections = self._select_diverse_subset(uncertain_candidates)
            self._copy_selected_files(final_selections)
        print("\nAll tasks completed.")

def main(args):
    ActiveLearningSampler(args).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Intelligently samples images for labeling using active learning.")
    parser.add_argument('--source', type=str, default=None, help="Relative path to the source dataset with the image pool.")
    parser.add_argument('--weights', type=str, default=None, help="Relative path to the teacher model (.pt) to be used for predictions.")
    parser.add_argument('--workdir', type=str, default=None, help="Relative path to the workspace where intermediate results will be saved.")
    parser.add_argument('--size', type=int, default=None, help="The final number of images to select.")
    parser.add_argument('--min_conf', type=float, default=None, help="Minimum confidence for uncertainty sampling.")
    parser.add_argument('--max_conf', type=float, default=None, help="Maximum confidence for uncertainty sampling.")
    parser.add_argument('--exist_ok', action='store_true', help="If set, overwrites the existing prediction directory.")
    args = parser.parse_args()
    main(args)