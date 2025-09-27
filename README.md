# YOLO Dataset Studio

Command-line workspace for curating YOLO datasets, supervising annotation loops, and training paired teacher/student models. Capture frames from ROS 2 bags or standard video, iterate labels in an integrated GUI, automate pseudo-labeling, and kick off active-learning cycles without stitching together ad-hoc scripts.

## Minimum Requirements
- Python 3.10 or later
- Ubuntu 20.04+/Debian/Windows/macOS with a POSIX-like shell (CLI tested primarily on Linux)
- GPU with recent CUDA drivers strongly recommended for YOLO training/inference (falls back to CPU)
- Optional GUI features require a desktop session with OpenCV window support
- Optional ROS 2 bag extraction requires a sourced ROS 2 distribution (e.g., Humble) and packages from `requirements-for-ros2bag.txt`

## Installation
```bash
pip install -r requirements.txt
# optional extras
pip install -r requirements-for-ros2bag.txt  # enable ROS 2 bag ingestion
```

For development work:
```bash
pip install -e .[dev]
# add ".[ros2bag]" if you need ROS 2 extras during development
```

## Configure Models & Workflow
- Edit `models_config.yaml` to define class IDs, YOLO architectures for teacher/student roles, training hyperparameters, and workflow parameters (image formats, auto-label thresholds, active-learning settings, etc.).
- All dataset paths are supplied interactively at runtime, so keep paths absolute when prompted to avoid confusion across shells.

## Launch the CLI
```bash
yolo-dataset-studio
# or, during development
python main.py
```

The CLI performs environment checks on startup and disables ROS 2 or GUI dependent features when the required libraries or display server are missing.

## Core Feature Areas
- **Data ingestion**: Bulk extraction from ROS 2 bags (non-interactive or GUI-assisted playback) and frame extraction from video files with single-shot or record-style capture.
- **Labeling toolkit**: Integrated OpenCV GUI with point-to-point drawing, magnifier window, hotkey-configurable classes, overlap diagnostics, review queues, isolation mode, and clipboard-style box reuse.
- **Training automation**: Launch Ultralytics YOLO training for teacher and student configs with progress monitoring, graceful quit-on-`q`, and run outputs under `runs/train/<role>/`.
- **Auto-labeling**: Apply any trained teacher checkpoint to new image pools with configurable batch sizes, writing YOLO-format labels beside discovered `images/` folders.
- **Dataset logistics**: Split datasets into train/val(/test) while generating `data.yaml`, merge multiple datasets via flatten or structure-preserving strategies, and sample subsets using random or uniform spacing.
- **Active learning**: Score unlabeled pools, filter uncertain predictions, embed with CLIP, select a diverse subset via K-Means, and copy candidates to a review folder for the next labeling sprint.

## CLI Menu Overview
| Option | Description |
| ------ | ----------- |
| 1 | Extract images from a ROS 2 bag (auto/interactive) |
| 2 | Extract frames from standard video files |
| 3 | Launch the integrated labeling GUI |
| 4 | Split a dataset and generate `data.yaml` |
| 5 | Train teacher or student YOLO models |
| 6 | Auto-label a dataset with a teacher checkpoint |
| 7 | Merge multiple datasets (flatten or structured) |
| 8 | Sample a dataset (random or uniform selection) |
| 9 | Register new dataset directories for this session |
| 0 | Exit |

## Typical Workflow Blueprint
1. Register or create a seed dataset (`9`, `1`, or `2`).
2. Label a representative subset in the GUI (`3`).
3. Split into train/val(/test) and produce `data.yaml` (`4`).
4. Train the teacher model (`5`).
5. Auto-label the remaining pool with the teacher checkpoint (`6`).
6. Review flagged images or refine labels (`3` with review filter).
7. Merge refined datasets or generate focused samples (`7`/`8`) and retrain (`5`).
8. Optional: run the active learning sampler for the next labeling sprint.

## Active Learning Sampler
The script `advanced_features/active_learning_sampler.py` orchestrates uncertainty filtering, CLIP feature extraction, and diversity sampling. Example usage:
```bash
python advanced_features/active_learning_sampler.py \
  --source path/to/unlabeled_images \
  --weights runs/train/teacher/<run>/weights/best.pt \
  --workdir active_learning_workspace \
  --size 100 \
  --exist_ok
```
Outputs:
- `predictions/` – YOLO predictions and confidences
- `features/` – CLIP embeddings per image
- `selected_for_labeling/` – curated subset to label next

## Project Layout (key pieces)
```
advanced_features/active_learning_sampler.py  # Active learning pipeline
main.py                                       # CLI entry point and menu routing
toolkit/data_handler.py                       # Extraction, splitting, merging, sampling utilities
toolkit/labeling.py                           # Integrated labeling GUI and auto-label helper
toolkit/training.py                           # YOLO training orchestration with progress callbacks
models_config.yaml                            # Classes, model configs, workflow defaults
examples/quickstart.py                        # Demo script to bootstrap a workspace
tests/test_smoke.py                           # Packaging smoke test
```

## Development & Testing
- Run smoke tests with `pytest`.
- `examples/quickstart.py` scaffolds an empty dataset directory and launches the CLI for first-time demos.
- Contributions should keep edits within ASCII unless an existing file uses otherwise.

