# End-to-End YOLO Dataset Pipeline with Semi-Supervised & Teacher-Student Learning

This is an end-to-end pipeline for building a custom YOLO dataset from images extracted from a ROS2 Bag. It progressively refines the dataset through **Semi-Supervised Learning** and a **Teacher-Student** architecture to train a final model.

The goal is to efficiently process large-scale, unstructured data collected in autonomous driving environments with minimal manual labeling, enabling the model to expand the dataset on its own and thereby maximize performance.

---

## Key Features

* **ROS2 Integration**: Directly extracts image frames from ROS2 Bags, unifying the process from data collection to model training.
* **Teacher-Student Architecture**: Employs an efficient strategy where a more complex Teacher model builds a high-quality dataset, which is then used to train a lighter or different architecture Student model.
* **Semi-Supervised Learning Cycle**: Provides an iterative workflow where a Teacher model, created from a small, manually labeled dataset, automatically labels new data. The user then reviews and corrects these labels to progressively improve both the dataset and the model's performance.
* **Intelligent Data Sampling**: Through Active Learning, the pipeline can intelligently select the most informative data for labeling to maximize performance gains.
* **Modular Workflow**: The entire pipeline is organized into three distinct phases and a set of auxiliary tools, making it easy to understand the project flow and selectively execute specific tasks.

---

## Project Structure
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

## Full Workflow Guide

This pipeline proceeds in three main phases.

### **Phase 1: Initial Teacher Model Construction (`1_Initial_Teacher_Workflow`)**

This is the starting point for the entire process, dedicated to creating the first "Teacher" model.

1.  **`1a_extract_from_bag.py`**: Extracts images from a ROS2 Bag file.
2.  **`1b_manual_labeler.py`**: Manually and accurately label a **small sample** of the extracted images.
3.  **`1c_split_for_training.py`**: Splits the labeled dataset into `train/val` sets and generates the necessary `data.yaml` file for training.
4.  **`1d_train_teacher_model.py`**: Trains the first **Teacher model** on the split dataset.

### **Phase 2: Semi-Supervised Cycle (`2_Semi_Supervised_Cycle`)**

This is the core iterative cycle that uses the Teacher model from Phase 1 to expand the dataset and improve performance.

1.  **`2a_auto_labeler.py`**: Uses the trained Teacher model to **automatically generate labels** for a large number of unlabeled images.
2.  **`2b_review_and_cleaner.py`**: Visually **reviews and corrects** the automatically generated labels to ensure data quality.
3.  **Merge and Retrain**:
    * Use `tools/merge_datasets.py` to combine the newly reviewed data with the existing dataset.
    * Run `1c_split_for_training.py` again to re-split the expanded dataset.
    * Execute `2c_retrain_model.py` to **retrain** the Teacher model with the larger, more refined dataset, thereby improving its performance.
4.  **Repeat this process** multiple times to build a large-scale, high-quality dataset.

### **Phase 3: Final Student Model Training (`3_Final_Student_Training`)**

This is the final stage, where the complete dataset created in Phase 2 is used to train the desired **Student model**.

1.  **Modify `_config.yaml`**: Specify the type of Student model to be trained (e.g., `yolov10n.pt`) and the path to the final dataset.
2.  **`3a_train_student_model.py`**: Trains the Student model on the final dataset to obtain a lightweight yet high-performance model.

---

## Installation and Setup

### **1. Clone the Repository**

```bash
git clone https://github.com/kdh10086/Creating-a-YOLO-Custom-Dataset-from-a-ros2bag-image_raw-Topic.git
cd Creating-a-YOLO-Custom-Dataset-from-a-ros2bag-image_raw-Topic
```

### 2. Install Dependencies

Install all required libraries using the `requirements.txt` file

```bash
pip install -r requirements.txt
```

`requirements.txt` Content:

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

*Note: ROS2-related libraries such as rosbag2_py and cv_bridge should already be part of your ROS2 development environment.*

### 3. Configure the Environment

All scripts reference the `_config.yaml` file in the root directory before execution. Please configure class information, dataset paths, model types, and hyperparameters to match your project environment.

```YAML
# Example _config.yaml

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

## Detailed Script Usage

Each script can be run from the terminal with command-line arguments. If arguments are omitted, the script will use the default settings from the `_config.yaml` file.

### **Phase 1: Initial Teacher Model Construction**

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

### **Phase 2: Semi-Supervised Cycle**

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

### **Phase 3: Final Student Model Training**

* **`3a_train_student_model.py`**
    ```bash
    python 3_Final_Student_Training/3a_train_student_model.py --dataset <final_dataset_dir>
    ```
    *Note: The Student model type is configured in _config.yaml.*

### **Tools**

* **`merge_datasets.py`**
    ```bash
    python tools/merge_datasets.py --inputs <dir1> <dir2> --output <merged_dir>
    ```

* **`sample_dataset.py`**
    ```bash
    python tools/sample_dataset.py --source <large_dataset_dir> --output <sample_dir> --ratio 0.1
    ```

### **Advanced Features**

* **`active_learning_sampler.py`**
    ```bash
    python advanced_features/active_learning_sampler.py --source <pool_dir> --weights <model.pt> --workdir <al_workspace> --size 200
    ```
