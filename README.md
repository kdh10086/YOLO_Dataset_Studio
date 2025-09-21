# ROS2 YOLO Toolkit

## 1. Overview

This project provides a comprehensive, end-to-end pipeline for creating custom YOLO models. It is designed to streamline the entire machine learning workflow, from data extraction and labeling to model training and semi-supervised learning.

The core of this toolkit is an interactive command-line interface (`main.py`) that guides the user through each step. The entire workflow is managed through this central tool and a comprehensive configuration file, `models_config.yaml`.

The toolkit supports a semi-supervised learning cycle. It begins by training an initial "Teacher" model on a small, manually-labeled dataset. This Teacher model then automatically labels a larger pool of unlabeled images. This iterative process efficiently builds a large-scale, high-quality dataset, which is finally used to train an optimized "Student" model for deployment.

## 2. Features

- **Interactive CLI**: A user-friendly command-line interface to manage the entire workflow.
- **ROS2 Bag Extraction**: Directly extract image frames from ROS2 bag files, with both automatic and interactive modes.
- **Integrated Labeling Tool**: A feature-rich GUI for manual labeling and reviewing, with point-to-point drawing, a zoom magnifier, and class selection.
- **Automated Data Handling**: Scripts for splitting datasets for training, merging multiple datasets, and creating random samples.
- **Teacher-Student Training**: Train a powerful Teacher model to supervise the training of a lightweight Student model.
- **Auto-Labeling**: Use a trained Teacher model to automatically generate pseudo-labels for a large set of unlabeled images.
- **Active Learning**: An advanced script to intelligently sample images for labeling based on model uncertainty and feature diversity.
- **Centralized Configuration**: Easily manage all paths, models, and hyperparameters from `models_config.yaml`.

## 3. Project Structure

```
.
├── main.py
├── models_config.yaml
├── requirements.txt
├── README.md
├── advanced_features/
│   └── active_learning_sampler.py
├── toolkit/
│   ├── data_handler.py
│   ├── labeling.py
│   ├── training.py
│   └── utils.py
├── datasets/
└── runs/
```

## 4. Prerequisites & Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For ROS2 bag extraction, a valid ROS2 installation (e.g., Humble) must be sourced in your environment.*

2.  **Configure `models_config.yaml`**: Before running the toolkit, open `models_config.yaml` and carefully set the required parameters. This includes model configurations, class names, and workflow parameters. The comments in the file explain the purpose of each variable.

## 5. Workflow and Usage Guide

The primary way to use this project is through the main interactive script.

```bash
python main.py
```

This will launch the ROS2 YOLO Toolkit, which provides a menu-driven interface to all the core functionalities.

### Main Menu Options

#### Data Preparation
- **[1] Extract Images from ROS Bag**: Extracts image frames from a ROS2 bag file. It can run in fully automatic mode (extracting all images) or interactive modes if a GUI is available.
- **[2] Launch Integrated Labeling Tool**: Starts a GUI tool for creating and reviewing bounding box labels. You can draw, delete, and modify labels, change classes, and manage a review list.
- **[3] Split Dataset for Training**: Takes a labeled dataset and splits it into `train` and `val` subsets, creating the `data.yaml` file required for training.

#### Training & Inference
- **[4] Train a Model (Teacher/Student)**: Trains a YOLO model. You can choose to train a 'Teacher' or a 'Student' model, and the script will use the corresponding configuration from `models_config.yaml`.
- **[5] Auto-label a Dataset with a Teacher**: Uses a trained Teacher model to automatically generate labels for a directory of unlabeled images.

#### Utilities
- **[6] Merge Datasets**: Combines multiple dataset directories into a single new dataset.
- **[7] Random Sample from Dataset**: Creates a smaller, randomly sampled subset from a larger dataset.
- **[8] Add New Dataset Directory**: Registers a new dataset path to be used within the toolkit session.

### General Workflow Example

1.  **Register Datasets**: Use option **[8]** to add the paths to your dataset directories.
2.  **Extract Images**: If you have ROS2 bags, use option **[1]** to extract images into a new dataset directory.
3.  **Label Initial Data**: Use option **[2]** to manually label a small, initial set of images.
4.  **Split for Training**: Use option **[3]** on your labeled dataset to prepare it for training.
5.  **Train First Teacher**: Use option **[4]** to train your first Teacher model on the small dataset.
6.  **Auto-label More Data**: Use option **[5]** with your new Teacher model to label a larger set of images.
7.  **Review & Refine**: Use the labeling tool **[2]** again to review and correct the auto-generated labels.
8.  **Merge and Retrain**: Use option **[6]** to merge your initial and newly labeled datasets. Then, train a new, improved Teacher model on this larger dataset.
9.  **Train Student Model**: Once you have a large, high-quality dataset, use option **[4]** to train the final, lightweight Student model for deployment.

## 6. Advanced Features

### Active Learning Sampler

For more efficient dataset growth, the `active_learning_sampler.py` script can be used to intelligently select images for labeling. Instead of random sampling, this tool selects images that the model is most uncertain about and that are visually diverse.

This helps to improve the model's performance by focusing labeling efforts on the most informative images.

**Usage:**

The script is run from the command line and takes several arguments.

```bash
python advanced_features/active_learning_sampler.py --source path/to/unlabeled_images --weights path/to/teacher_model.pt --workdir path/to/workspace --size 100
```

- `--source`: Path to the directory with the large pool of unlabeled images.
- `--weights`: Path to the trained Teacher model weights.
- `--workdir`: A directory to store intermediate files (predictions, features).
- `--size`: The final number of images to select for labeling.

The selected images will be copied to the `selected_for_labeling` sub-directory inside your specified workspace.

## 7. Configuration

The `models_config.yaml` file is the central place to manage all settings.

- **`model_configurations`**: Define the classes, and configure your Teacher and Student models, including the model architecture (`model_name`) and training hyperparameters (`epochs`, `batch_size`, `img_size`).
- **`workflow_parameters`**: Control the behavior of the toolkit scripts, such as the ROS topic for image extraction, the train/validation split ratio, and the confidence threshold for auto-labeling.