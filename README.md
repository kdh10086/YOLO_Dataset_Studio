# YOLO Dataset Studio

Integrated CLI workspace for curating custom YOLO datasets, automating labeling loops, and training teacher/student models. Bonus: extract image datasets straight from ROS2 bag files when you need them.

## Why this studio?
Robotics and vision teams often juggle multiple scripts to source data, clean labels, train models, and re-label with fresh checkpoints. YOLO Dataset Studio bundles that workflow into a single command-line experience so you can focus on iterating datasets for YOLO models instead of wiring together ad-hoc utilities. ROS2 bag conversion is built in, but the main intent is broader: manage any YOLO-friendly dataset end-to-end.

## Core capabilities
### Data acquisition & sourcing
- Register any dataset directory once and reuse it across the session.
- Extract frames from ROS2 bags in bulk or via interactive playback with pause/save controls.
- Create quick samples from large datasets for smoke testing or labeling sprints.

### Labeling & review
- Launch a feature-rich GUI labeler with point-to-point box drawing, zoom magnifier, class hotkeys, review lists, and the ability to sideline problematic frames.
- Auto-label entire datasets using a trained Teacher checkpoint and configurable confidence thresholds.

### Training & automation
- Train Teacher and Student YOLO models with unified progress reporting and graceful interrupt handling.
- Iterate semi-supervised cycles: kick off training, auto-label unlabeled pools, then refine the annotations in the GUI.

### Dataset logistics
- Split datasets into train/val(/test) with flexible directory layouts and automatic `data.yaml` generation.
- Merge multiple datasets via flatten or structure-preserving strategies, keeping labels in sync.
- Centralize class definitions, paths, and hyperparameters inside `models_config.yaml` so experiments stay reproducible.

## Bonus: ROS2 bag integration
Running `python main.py` automatically checks whether ROS2 dependencies and a GUI are available. If ROS2 is sourced, you can:
- Perform fast offline extraction through `rosbag2_py`.
- Drive an interactive `ros2 bag play` session, pause via services, and save frames on demand.
Make sure a ROS2 distribution (e.g., Humble) is installed and sourced before launching the studio.

## Project layout
```
.
├── main.py                     # Interactive CLI entry point
├── models_config.yaml          # Central configuration for models & workflow
├── requirements.txt
├── advanced_features/
│   └── active_learning_sampler.py
├── toolkit/
│   ├── data_handler.py         # Dataset ops, bag extraction, splits, merges
│   ├── labeling.py             # GUI labeler and auto-label helpers
│   ├── training.py             # YOLO training orchestration
│   └── utils.py
├── datasets/                   # Default workspace for generated datasets
└── runs/                       # YOLO training outputs (Ultralytics format)
```

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Optional: source your ROS2 setup if you plan to work with bag files.
2. Configure `models_config.yaml`: set class names, dataset roots, YOLO model variants, and workflow parameters (topics, split ratios, confidence thresholds, etc.).
3. Launch the CLI:
   ```bash
   python main.py
   ```
   The menu adapts to your environment, disabling ROS2 or GUI features when unavailable.

## Interactive CLI at a glance
- **[1] Extract Images from ROS Bag** – Bulk or interactive playback extraction into YOLO-ready folders.
- **[2] Launch Integrated Labeling Tool** – Full-screen GUI with review queues, isolation, and class shortcuts.
- **[3] Split Dataset for Training** – Train/val(/test) splits with directory layout selection and `data.yaml` creation.
- **[4] Train a Model (Teacher/Student)** – Executes Ultralytics training using configs, with live progress bars.
- **[5] Auto-label a Dataset with a Teacher** – Runs inference over image pools and writes YOLO-format labels.
- **[6] Merge Datasets** – Combine projects via flatten or structure-preserving strategies.
- **[7] Sample from Dataset** – Build quick subsets by random sampling pairs.
- **[8] Add New Dataset Directory** – Register additional dataset roots on the fly.

## Semi-supervised workflow blueprint
1. Register or create a seed dataset (Options 8 and/or 1).
2. Label a high-quality subset manually (Option 2).
3. Split and generate `data.yaml` (Option 3).
4. Train the initial Teacher model (Option 4).
5. Auto-label the remaining pool (Option 5).
6. Review and correct Teacher labels (Option 2 with review mode).
7. Merge refined datasets and retrain Teachers/Students as needed (Options 6 and 4).

## Active learning sampler
`advanced_features/active_learning_sampler.py` scores unlabeled images using your Teacher model and selects a diverse subset for manual labeling. Example:
```bash
python advanced_features/active_learning_sampler.py \
  --source path/to/unlabeled_images \
  --weights path/to/teacher_model.pt \
  --workdir path/to/workspace \
  --size 100
```
Outputs are stored under `selected_for_labeling/` inside the workspace directory.

## Configuration tips
- `model_configurations` groups Teacher/Student settings, including separate hyperparameters per model variant.
- `workflow_parameters` cover ROS topics, output formats, auto-label thresholds, and split ratios.
- Keep dataset paths absolute to avoid confusion when launching from different shells.

## Outputs and artifacts
- Datasets you create or import live under `datasets/` (or any paths you register).
- YOLO training runs follow the Ultralytics convention in `runs/train/<role>/<run_name>/` with metrics and checkpoints.
- Auto-labeling writes directly next to the source images, respecting `images/` → `labels/` folder patterns.
