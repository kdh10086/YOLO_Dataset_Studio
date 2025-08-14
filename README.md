# End-to-End YOLO Model Training Pipeline with Semi-Supervised Learning

## 1. Overview

This project provides a comprehensive, end-to-end pipeline for creating a custom, lightweight YOLO "Student" model tailored for specific environments with limited computing resources.

The core strategy is **bootstrapping** via a semi-supervised learning cycle. It begins by training an initial "Teacher" model on a small, manually-labeled dataset. This Teacher model then automatically labels a larger pool of unlabeled images. After human review and correction, this newly labeled data is used to retrain a more robust Teacher model. This iterative process efficiently builds a large-scale, high-quality dataset, which is finally used to train the optimized, lightweight Student model for deployment.

The entire workflow is managed through a series of modular scripts and a central configuration file, `_config.yaml`.

---

## 2. Features

- **ROS2 Bag Extraction**: Directly extract image frames from ros2bag files.
- **Manual & Auto Labeling**: Includes a feature-rich GUI for manual labeling and a script for automated pseudo-labeling.
- **Data Review & Cleaning**: Tools for visually inspecting labels, marking images for review, and cleaning noisy data.
- **Semi-Supervised Cycle**: An iterative workflow to progressively improve the model and dataset size.
- **Teacher-Student Architecture**: Train a powerful Teacher model to supervise the training of a lightweight Student model.
- **Centralized Configuration**: Easily manage all paths, models, and hyperparameters from `_config.yaml`.

---

## 3. Project Structure
```
.
├── 1_Initial_Teacher_Workflow/
│   ├── 1a_extract_from_bag.py
│   ├── 1b_manual_labeler.py
│   ├── 1c_view_simple_labels.py
│   ├── 1d_split_for_training.py
│   └── 1e_train_teacher_model.py
├── 2_Semi_Supervised_Cycle/
│   ├── 2a_auto_labeler.py
│   ├── 2b_review_and_cleaner.py
│   └── 2c_retrain_model.py
├── 3_Final_Student_Training/
│   └── 3a_train_student_model.py
├── advanced_features/
│   └── active_learning_sampler.py
├── tools/
│   ├── merge_datasets.py
│   └── random_sample_dataset.py
├── datasets/
├── runs/
├── _config.yaml
├── requirements.txt
└── README.md
```

---

## 4. Prerequisites & Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure `_config.yaml`**: Before running any scripts, open `_config.yaml` and carefully set all required paths (especially `ros2bag_directory`) and parameters. The comments in the file explain the purpose of each variable.

---

## 3. Workflow and Detailed Usage Guide

This section provides a step-by-step guide on how to use the package to create your custom model.

### Phase 1: Initial Dataset and First Teacher Model

The goal of this phase is to create a small, high-quality, manually-labeled dataset and use it to train an initial "Teacher" model.

* **Step 1.1: Extract Initial Frames**
    * **Goal**: Select a small, diverse set of images to serve as the initial seed for your dataset.
    * **Action**: Run `1a_extract_from_bag.py` to extract approximately 200 dynamically changing frames from your ROS bag file. Save them to your initial dataset directory (e.g., `datasets/sample_dataset/images`).

* **Step 1.2: Manual Labeling**
    * **Goal**: Manually label the initial set of images.
    * **Action**: Run `1b_manual_labeler.py`. Set the target dataset in the script or via command-line arguments. Carefully draw bounding boxes and assign classes to all objects in the images.

* **Step 1.3: Review and Refine**
    * **Goal**: Check the manually labeled data for errors and correct them.
    * **Action**: Run `1c_view_simple_labels.py` to visually inspect your work. If you find an image with an error, left-click on it. This adds the image's filename to a `review.txt` file. Once you have a list of files to fix, run `1b_manual_labeler.py` again. The script can be started in a "review mode" to load only the images from `review.txt` for correction. Repeat this process until your dataset is accurate.

* **Step 1.4: Prepare for Training**
    * **Goal**: Structure the dataset correctly for YOLO training.
    * **Action**: Run `1d_split_for_training.py` on your verified dataset directory. This script will automatically split your images and labels into `train/` and `val/` subdirectories and generate the necessary `data.yaml` file.

* **Step 1.5: Train the First Teacher Model**
    * **Goal**: Train the first iteration of the Teacher model.
    * **Action**: Run `1e_train_teacher_model.py`. Ensure the `dataset_paths` in `_config.yaml` point to your prepared dataset. This will produce the first set of model weights (e.g., `best.pt`), which are essential for the next phase.

### Phase 2: Bootstrapping and Improving the Teacher

The goal here is to use the first Teacher model to label more data, then use that expanded dataset to train a better Teacher model.

* **Step 2.1: Create a Large Unlabeled Pool**
    * **Goal**: Extract all available frames to create a large pool of data for auto-labeling.
    * **Action**: Run `1a_extract_from_bag.py` again, but this time, extract all frames from the bag file into a new directory for your large-scale dataset.

* **Step 2.2: Create a Medium-Sized Sample**
    * **Goal**: Select a manageable subset of the large pool for the first bootstrapping cycle.
    * **Action**: Run `tools/random_sample_dataset.py`. Point it to the large dataset directory created in the previous step and generate a random sample of approximately 400 images.

* **Step 2.3: Auto-Label the Sampled Data**
    * **Goal**: Use the first Teacher model to automatically generate labels for the 400-image dataset.
    * **Action**: In `_config.yaml`, update the `semi_supervised_weights` path to point to the `best.pt` file from Phase 1. Then, run `2a_auto_labeler.py` on the 400-image dataset.

* **Step 2.4: Review the Auto-Labeled Data**
    * **Goal**: Clean and correct the pseudo-labels generated by the Teacher model.
    * **Action**: Use `2b_review_and_cleaner.py` to inspect the auto-labeled results. You can mark entire frames for deletion or add them to a `review_list.txt` for fine-tuning. If corrections are needed, use `1b_manual_labeler.py` (in review mode) to fix the labels. Use `1c_view_simple_labels.py` as needed to verify.

* **Step 2.5: Merge Datasets**
    * **Goal**: Combine the initial manually labeled data with the new, cleaned auto-labeled data.
    * **Action**: Run `tools/merge_datasets.py`. Set the input directories to your initial dataset (~200 images) and the cleaned auto-labeled dataset (~400 images). This creates a new, larger `merged_dataset`.

* **Step 2.6: Prepare and Retrain the Teacher**
    * **Goal**: Train a new, improved Teacher model on the larger, combined dataset.
    * **Action**: First, run `1d_split_for_training.py` on the `merged_dataset`. Then, run `2c_retrain_model.py`. You may consider using a model architecture with more parameters (e.g., YOLOv8m instead of YOLOv8n) in `_config.yaml` to create a more powerful Teacher.

### Phase 3: Final Dataset Creation and Student Model Training

This final phase uses the improved Teacher model to create the full, high-quality dataset and train the final, lightweight Student model.

* **Step 3.1: Large-Scale Auto-Labeling**
    * **Goal**: Label the entire pool of extracted images using the improved Teacher model.
    * **Action**: In `_config.yaml`, update the `semi_supervised_weights` path to the `best.pt` file from the *retrained* Teacher in Phase 2. Run `2a_auto_labeler.py` on the full dataset extracted in Step 2.1.

* **Step 3.2: Final Review**
    * **Goal**: Ensure the final large-scale dataset is as clean as possible.
    * **Action**: Perform a final review and correction pass on the entire dataset using `2b_review_and_cleaner.py` and `1b_manual_labeler.py` as needed.

* **Step 3.3: Train the Final Student Model**
    * **Goal**: Train the final, small, and efficient model for deployment.
    * **Action**: In `_config.yaml`, configure the `student_model_config` section, choosing a lightweight architecture (e.g., `yolov10s`). Ensure `final_training_dataset` points to your completed large-scale dataset. Run `3a_train_student_model.py`.

* **Step 3.4: Deployment**
    * **Goal**: Use the trained model.
    * **Action**: The resulting Student model (`best.pt` from the final training run) is now optimized for your specific environment and ready for deployment.

* **Step 3.5: Post-Training Robustness Improvement (Optional)**
    * **Goal**: Enhance the model's robustness if it performs poorly on new, unseen data.
    * **Action**: Test the trained Student model on a new dataset created from a different ROS bag file. If the model shows weakness in certain environments, use `advanced_features/active_learning_sampler.py` on the new data. This will intelligently select additional images from the weak environments. Merge these newly selected images with your existing large-scale dataset and retrain the Student model. This process can significantly improve the model's generalization and robustness.

---

## 6. Script Descriptions

### 1_Initial_Teacher_Workflow
- `1a_extract_from_bag.py`: Extracts image frames from a ROS2 bag file.
- `1b_manual_labeler.py`: A GUI tool for drawing, deleting, and editing bounding box labels.
- `1c_view_simple_labels.py`: A simple viewer to display images with their YOLO labels and manage a review list.
- `1d_split_for_training.py`: Splits a dataset into training/validation sets and generates a `data.yaml` file.
- `1e_train_teacher_model.py`: Trains the initial YOLO Teacher model.

### 2_Semi_Supervised_Cycle
- `2a_auto_labeler.py`: Uses a pre-trained Teacher model to automatically generate labels for unlabeled images.
- `2b_review_and_cleaner.py`: A GUI tool to visualize auto-labeled data, mark noisy frames for removal, or flag images for manual review.
- `2c_retrain_model.py`: Retrains the Teacher model on a larger, combined dataset.

### 3_Final_Student_Training
- `3a_train_student_model.py`: Trains the final, lightweight Student model.

### advanced_features
- `active_learning_sampler.py`: Intelligently selects a diverse subset of images for labeling from a large pool based on model uncertainty and feature clustering.

### tools
- `merge_datasets.py`: Combines multiple dataset folders into a single, new dataset.
- `random_sample_dataset.py`: Creates a smaller, randomly sampled subset from a larger dataset.

---

## 7. General Usage

There are two primary ways to run the scripts in this project: from the command line or directly within a VS Code environment.

### Method 1: Command Line Interface

Each script can be run from your terminal. This method is flexible as it allows you to override any default setting from `_config.yaml` by passing command-line arguments.

For example, to run the auto-labeler on a specific dataset with a custom confidence threshold:
```bash
# Run the auto-labeler with custom arguments
python 2_Semi_Supervised_Cycle/2a_auto_labeler.py --dataset path/to/my/dataset --weights path/to/model.pt --conf 0.4
```

To see all available arguments for any script, use the `--help` flag:
```bash
python 1a_extract_from_bag.py --help
```

### Method 2: VS Code with Code Runner

For convenience during development, scripts can be run directly within the VS Code editor using an extension like Code Runner. When using this method, script parameters are determined by the following priority:

1.  **Initiation Settings**: You can directly edit the `INIT_...` variables within the "Initiation Settings" section at the top of each script file for a quick change.
2.  **`_config.yaml`**: If the `INIT_...` variables are left as `None`, the script will automatically load and use the corresponding values from the central `_config.yaml` file.

This allows for rapid testing and execution without needing to type command-line arguments for each run.
