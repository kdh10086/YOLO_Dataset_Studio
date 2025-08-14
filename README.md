# End-to-End YOLO Dataset Pipeline with Semi-Supervised & Teacher-Student Learning

ROS2 Bag에서 추출한 이미지로 YOLO 커스텀 데이터셋을 구축하고, **준지도 학습(Semi-Supervised Learning)** 및 **Teacher-Student** 아키텍처를 통해 데이터셋을 점진적으로 고도화하며 최종 모델을 학습시키는 End-to-End 파이프라인입니다.

자율주행 환경에서 수집된 대규모의 비정형 데이터를 최소한의 수동 라벨링으로 효율적으로 가공하고, 모델 스스로 데이터셋을 확장하게 하여 성능을 극대화하는 것을 목표로 합니다.



---

## 주요 특징

* **ROS2 연동**: ROS2 Bag에서 이미지 프레임을 직접 추출하여 데이터 수집부터 학습까지의 과정을 통합합니다.
* **Teacher-Student 구조**: 무거운 Teacher 모델로 고품질의 데이터셋을 구축하고, 최종적으로 가볍거나 다른 아키텍처의 Student 모델을 학습시키는 효율적인 전략을 사용합니다.
* **준지도 학습 사이클**: 초기에 수동으로 라벨링한 소수의 데이터로 Teacher 모델을 만든 후, 이 모델이 나머지 데이터를 자동으로 라벨링하고, 사용자는 그 결과를 검토/수정하여 점진적으로 데이터셋과 모델 성능을 향상시키는 반복적 워크플로우를 제공합니다.
* **지능형 데이터 샘플링**: Active Learning을 통해 모델 성능 향상에 가장 도움이 될 만한 데이터를 지능적으로 선별하여 라벨링을 요청할 수 있습니다.
* **모듈화된 워크플로우**: 전체 파이프라인이 명확한 3단계와 보조 도구들로 구성되어 있어, 프로젝트의 흐름을 쉽게 이해하고 필요한 기능을 선택적으로 사용할 수 있습니다.

---

## 프로젝트 구조
```
.
├── 1_Initial_Teacher_Workflow/
│   ├── 1a_extract_from_bag.py
│   ├── 1b_manual_labeler.py
│   ├── 1c_split_for_training.py
│   └── 1d_train_teacher_model.py
├── 2_Semi_Supervised_Cycle/
│   ├── 2a_auto_labeler.py
│   ├── 2b_review_and_cleaner.py
│   └── 2c_retrain_model.py
├── 3_Final_Student_Training/
│   └── 3a_train_student_model.py
├── advanced_features/
│   └── active_learning_sampler.py
├── tools/
│   ├── view_simple_labels.py
│   ├── merge_datasets.py
│   └── sample_dataset.py
├── datasets/
├── runs/
├── _config.yaml
├── requirements.txt
└── README.md
```

---

## 전체 워크플로우 가이드

이 파이프라인은 크게 3가지 단계로 진행됩니다.

### **Phase 1: 초기 Teacher 모델 구축 (`1_Initial_Teacher_Workflow`)**

모든 과정의 시작점으로, 첫 번째 '선생님 모델'을 만들기 위한 단계입니다.

1.  **`1a_extract_from_bag.py`**: ROS2 Bag 파일에서 이미지들을 추출합니다.
2.  **`1b_manual_labeler.py`**: 추출된 이미지 중 **소량의 샘플**에 대해 수동으로 정확하게 라벨링합니다.
3.  **`1c_split_for_training.py`**: 라벨링된 데이터셋을 `train/val`로 분할하고 학습에 필요한 `data.yaml` 파일을 생성합니다.
4.  **`1d_train_teacher_model.py`**: 분할된 데이터셋으로 첫 번째 **Teacher 모델**을 학습시킵니다.

### **Phase 2: 준지도 학습 사이클 (`2_Semi_Supervised_Cycle`)**

1단계에서 만든 Teacher 모델을 이용해 데이터셋을 확장하고 모델 성능을 반복적으로 고도화하는 핵심 사이클입니다.

1.  **`2a_auto_labeler.py`**: 1단계에서 학습된 Teacher 모델을 사용하여 아직 라벨링되지 않은 대량의 이미지에 대해 **자동으로 라벨을 생성**합니다.
2.  **`2b_review_and_cleaner.py`**: 자동 라벨링된 결과물을 시각적으로 확인하며, 잘못된 부분을 **검토하고 수정**합니다. 이 과정을 통해 데이터의 품질을 높입니다.
3.  **데이터 병합 및 재학습**:
    * `tools/merge_datasets.py`를 사용하여, 검수가 완료된 새로운 데이터와 기존 데이터를 하나로 합칩니다.
    * 다시 `1c_split_for_training.py`를 실행하여 확장된 데이터셋을 분할합니다.
    * `2c_retrain_model.py`를 실행하여 더 많고 정교해진 데이터로 Teacher 모델을 **재학습**시켜 성능을 향상시킵니다.
4.  **이 과정을 여러 번 반복**하여 대규모의 고품질 데이터셋을 완성합니다.

### **Phase 3: 최종 Student 모델 학습 (`3_Final_Student_Training`)**

2단계를 통해 완성된 최종 데이터셋을 사용하여, 우리가 실제로 사용하고자 하는 **Student 모델**을 학습시키는 마지막 단계입니다.

1.  **`_config.yaml` 수정**: 학습시킬 Student 모델의 종류(예: `yolov10n.pt`)와 최종 데이터셋 경로를 지정합니다.
2.  **`3a_train_student_model.py`**: 최종 데이터셋으로 Student 모델을 학습시켜, 가벼우면서도 높은 성능을 가진 최종 모델을 얻습니다.

---

## 설치 및 설정

### **1. 저장소 복제**

```bash
git clone [https://github.com/Doooo-Hyeong/Creating-a-YOLO-Custom-Dataset.git](https://github.com/Doooo-Hyeong/Creating-a-YOLO-Custom-Dataset.git)
cd Creating-a-YOLO-Custom-Dataset
```

### 2. 필요 라이브러리 설치

`requirements.txt` 파일을 이용하여 필요한 모든 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

`requirements.txt` 내용:

```Plaintext
# Main Machine Learning and Vision Libraries
ultralytics
torch
torchvision
opencv-python
numpy

# Data Handling and Utility Libraries
pyyaml
tqdm
Pillow
scikit-learn

# Hugging Face for CLIP Model (used in Active Learning)
transformers

# Matplotlib for plotting (dependency of ultralytics)
matplotlib
```

참고: `rosbag2_py`, `cv_bridge` 등 ROS2 관련 라이브러리는 사용자의 ROS2 개발 환경에 이미 포함되어 있어야 합니다.

### 3. 환경 설정

모든 스크립트는 실행 전 루트 디렉토리의 `_config.yaml` 파일을 참조합니다. 자신의 프로젝트 환경에 맞게 클래스 정보, 데이터셋 경로, 모델 종류, 하이퍼파라미터 등을 먼저 설정해주세요.

```YAML
# _config.yaml 예시

project_settings:
  classes:
    0: 'car'
    1: 'person'
  image_format: 'png,jpg'

initial_teacher_workflow:
  ros2_bag_extraction:
    directory: '/path/to/your/rosbags'
    image_topic: '/image_raw'
    output_dir: 'datasets/phase1_extracted'
  initial_labeled_dataset: 'datasets/phase1_initial_labeled'
  train_ratio: 0.8
  teacher_model_config:
    model_name: 'yolov10m'
    # ...
```

## 스크립트 상세 사용법

각 스크립트는 터미널에서 인자(argument)를 직접 전달하여 실행할 수 있습니다. 인자를 생략하면 `_config.yaml` 파일의 기본 설정을 따릅니다.

### **Phase 1: 초기 Teacher 모델 구축**

* **`1a_extract_from_bag.py`**
    ```bash
    python 1_Initial_Teacher_Workflow/1a_extract_from_bag.py --bag <bag_dir> --output <output_dir> --mode <1 or 2>
    ```

* **`1b_manual_labeler.py`**
    ```bash
    python 1_Initial_Teacher_Workflow/1b_manual_labeler.py --dataset <dataset_dir> --start_image <image_name.png>
    ```

* **`1c_split_for_training.py`**
    ```bash
    python 1_Initial_Teacher_Workflow/1c_split_for_training.py --dataset <dataset_dir>
    ```

* **`1d_train_teacher_model.py`**
    ```bash
    python 1_Initial_Teacher_Workflow/1d_train_teacher_model.py --dataset <dataset_dir> --epochs 100 --batch 16 --imgsz 640
    ```

### **Phase 2: 준지도 학습 사이클**

* **`2a_auto_labeler.py`**
    ```bash
    python 2_Semi_Supervised_Cycle/2a_auto_labeler.py --dataset <unlabeled_dir> --weights <teacher_model.pt> --conf 0.25
    ```

* **`2b_review_and_cleaner.py`**
    ```bash
    python 2_Semi_Supervised_Cycle/2b_review_and_cleaner.py --dataset <auto_labeled_dir> --mode remove_noise --output_dir <cleaned_dir>
    ```

* **`2c_retrain_model.py`**
    ```bash
    python 2_Semi_Supervised_Cycle/2c_retrain_model.py --dataset <merged_and_resplit_dir> --epochs 150
    ```

### **Phase 3: 최종 Student 모델 학습**

* **`3a_train_student_model.py`**
    ```bash
    python 3_Final_Student_Training/3a_train_student_model.py --dataset <final_dataset_dir>
    ```
    *참고: Student 모델 종류는 `_config.yaml`에서 수정합니다.*

### **Tools (보조 도구)**

* **`merge_datasets.py`**
    ```bash
    python tools/merge_datasets.py --inputs <dir1> <dir2> --output <merged_dir>
    ```

* **`sample_dataset.py`**
    ```bash
    python tools/sample_dataset.py --source <large_dataset_dir> --output <sample_dir> --ratio 0.1
    ```

### **Advanced Features (고급 기능)**

* **`active_learning_sampler.py`**
    ```bash
    python advanced_features/active_learning_sampler.py --source <pool_dir> --weights <model.pt> --workdir <al_workspace> --size 200
    ```
