# ML_comp

Detection of 25 object classes in optical satellite imagery under limited-data and class-imbalance conditions.

The current reference model is RT-DETR-L pretrained on COCO. Baseline v1 is complete and preserved as the historical project reference. Baseline v2 will focus on Fair Training, Copy-Paste, and Scene Mixing.

## Current Status

- The official dataset contains 4,481 images and 4,481 YOLO annotation files.
- The organizer-provided data is stored unchanged in `data/raw`.
- The original baseline v1 split has been reconstructed and fixed as manifests with 3,585 training images and 896 validation images.
- Baseline v1 has been evaluated on all 896 validation images.
- Baseline v2 will use its own versioned split and controlled ablation experiments.

At the selected baseline v1 operating point (`confidence=0.50`, matching IoU `0.50`, class-aware NMS IoU `0.50`), the full validation set gives Recall `0.903`, Precision `0.824`, and FAR `0.176`. The standard Ultralytics evaluation gives mAP50 `0.801` and mAP50–95 `0.449`.

## Repository Structure

```text
ML_comp/
├── configs/       # Dataset, training, and inference configurations
├── data/          # Unchanged source data and fixed split manifests
├── docker/        # Competition Docker image
├── docs/          # Proposal, dataset description, and experiment reports
├── models/        # Pretrained and selected trained weights
├── notebooks/     # EDA, evaluation, plots, and error analysis
├── runs/          # Experiment outputs and validation artifacts
├── scripts/       # Data preparation, training, evaluation, and inference entry points
└── src/ml_comp/   # Reusable project logic
```

- `configs/` stores shared experiment parameters.
- `data/raw/` stores the unchanged official dataset.
- `data/splits/` stores versioned train/validation manifests tracked by Git.
- `models/pretrained/` stores original pretrained weights.
- `models/trained/` stores selected team checkpoints.
- `runs/` stores experiment metrics, plots, and temporary outputs.
- `notebooks/` is used for analysis and validation, not for dataset mutation.
- `docker/` will package the final competition inference solution.

## Repository Guide

| Directory | Purpose | Documentation |
|---|---|---|
| `configs/` | Dataset and experiment configurations | [Details](configs/README.md) |
| `data/` | Source data and versioned split manifests | [Details](data/README.md) |
| `docker/` | Competition inference container | [Details](docker/README.md) |
| `docs/` | Project documents and experiment reports | [Details](docs/README.md) |
| `models/` | Pretrained and selected trained checkpoints | [Details](models/README.md) |
| `notebooks/` | Analysis, evaluation, and visualization | [Details](notebooks/README.md) |
| `runs/` | Generated experiment outputs | [Details](runs/README.md) |
| `scripts/` | Command-line entry points | [Details](scripts/README.md) |
| `src/` | Reusable Python implementation | [Details](src/README.md) |

Split-specific rules and naming conventions are documented in [data/splits/README.md](data/splits/README.md).

## Data Splits

The official dataset contains only a training subset. Baseline v1 uses the reconstructed historical split:

```text
data/splits/baseline_v1/
├── train.txt       # 3,585 images
└── val.txt         # 896 images
```

The validation manifest reproduces the historical validation class fingerprint. Ultralytics removes one duplicated annotation during evaluation, leaving 4,294 ground-truth objects.

Baseline v2 must use a separate directory such as `data/splits/baseline_v2/`. A source scene must not be shared between training and validation, and rare classes must remain represented in training.

## Data and Large Files

The dataset and experiment outputs are shared as versioned archives rather than committed to GitHub. Curated checkpoints under `models/trained/original_labels/` and `models/trained/new_labels/` are tracked with Git LFS:

```text
ml_comp_dataset_v1.0.tar.gz
ml_comp_models_v1.0.tar.gz
ml_comp_run_baseline_v2_v1.0.tar.gz
```

The dataset archive must produce this layout:

```text
data/raw/
├── images/
│   └── train/
└── labels/
    └── train/
```

The selected baseline v1 checkpoint is stored as:

```text
models/trained/baseline_v1_best.pt
```

## Environment

```bash
conda create -n ML_comp python=3.13 -y
conda activate ML_comp

python -m pip install \
  torch==2.11.0+cu128 \
  torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
```

Environment check:

```bash
python -c "import torch, ultralytics; print(torch.__version__); print(torch.cuda.is_available()); print(ultralytics.__version__)"
```

## Configurations and Commands

`configs/baseline_v1/dataset.yaml` defines the fixed baseline v1 manifests and 25 classes. `configs/baseline_v1/baseline_v1.yaml` records the experiment parameters.

The standalone training and evaluation scripts are currently placeholders. Do not treat a command as implemented while its script is empty.

Inference with the selected checkpoint:

```bash
python scripts/predict.py \
  --weights models/trained/baseline_v1_best.pt \
  --source /path/to/images
```

Full validation and threshold analysis are currently performed in:

```text
notebooks/baseline_v1/baseline_v1_results.ipynb
```

## Working Rules

1. Never modify or move files in `data/raw`.
2. Never create or modify a train/validation split inside an analysis notebook.
3. Never overwrite a finalized split; create a new version instead.
4. Do not commit datasets, unselected `.pt` files, run directories, or archives to Git. Only curated checkpoints under `models/trained/original_labels/` and `models/trained/new_labels/` may be tracked through Git LFS.
5. Save the configuration, aggregate metrics, per-class metrics, and best checkpoint for every experiment.
6. Compare model ablations only on the same fixed split and with the same evaluation protocol.

## Documents

- `docs/general/proposal.pdf` — original proposal.
- `docs/general/project_context.pdf` — project and competition description.
- `docs/general/dataset_description.md` — classes and YOLO annotation format.
- `notebooks/baseline_v1/baseline_v1_analysis.ipynb` — historical baseline v1 analysis.
- `notebooks/baseline_v1/baseline_v1_results.ipynb` — full validation, threshold sweep, per-class metrics, and runtime measurements.
- `docs/baseline_v1/baseline_v1_results.md` — baseline v1 results, limitations, and competition requirements.
