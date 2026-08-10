# Scripts

This directory contains command-line entry points for project workflows.

```text
scripts/
├── dataset_inventory.py  # Snapshot source files and class statistics
├── validate_dataset.py   # Validate images and YOLO annotations
├── prepare_data.py    # Build and validate split manifests
├── train.py           # Train a configured experiment
├── evaluate.py        # Evaluate a checkpoint on a fixed split
└── predict.py         # Run inference on images
```

Scripts should remain thin: parse arguments, load configuration, call reusable code from `src/ml_comp/`, and write outputs to an explicit run directory.

An entry point is not considered implemented while its file is empty. Commands and defaults must use repository-relative paths and must not modify `data/raw`.

Create the initial immutable-dataset inventory for a manual audit:

```bash
python scripts/dataset_inventory.py --output runs/dataset_audit/v1
```

The command only reads `data/raw` and writes `summary.json`, `file_inventory.csv`,
and `class_statistics.csv` to the selected output directory.

Run automatic integrity and annotation checks:

```bash
python scripts/validate_dataset.py --output runs/dataset_audit/v1
```
