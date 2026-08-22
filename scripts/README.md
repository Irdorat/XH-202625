# Scripts

This directory contains command-line entry points for project workflows.

```text
scripts/
├── dataset_inventory.py  # Snapshot source files and class statistics
├── validate_dataset.py   # Validate images and YOLO annotations
├── remap_shiprsimage.py  # Remap ShipRSImageNet to HM/LQS/QHS/MS
├── remap_mar20.py        # Remap MAR20 to project aircraft IDs 4..23
├── find_image_duplicates.py # Compare datasets using exact and perceptual hashes
├── prepare_data.py    # Build and validate split manifests
├── train.py           # Train a configured experiment
├── evaluate.py        # Evaluate a checkpoint on a fixed split
├── predict.py         # Run inference on images
└── sliced_predict.py  # Run tiled inference on one large image
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

Create a strict four-class ShipRSImageNet derivative without modifying the source dataset:

```bash
python scripts/remap_shiprsimage.py
```

Create a one-variant-per-source MAR20 derivative with project class IDs:

```bash
python scripts/remap_mar20.py
```

Audit MAR20 for overlap with the organizer dataset:

```bash
python scripts/find_image_duplicates.py
```

Evaluate all curated checkpoints on the matching fold manifests and regenerate
the aggregate and per-class reports:

```bash
python scripts/evaluate.py \
  --original-data-dir /path/to/original-label-data \
  --new-data-dir /path/to/new-label-data \
  --output-dir models/trained/metrics \
  --work-dir runs/checkpoint_per_class_metrics \
  --image-size 800 \
  --batch-size 64 \
  --workers 16 \
  --device 0 \
  --framework-revision SOURCE_ID
```

The command reads all curated weights, validates each checkpoint against its
own label version and fold, and writes one resumable cache per checkpoint.
The committed report schema and metric definitions are documented in
[`models/trained/metrics/`](../models/trained/metrics/README.md).

Run end-to-end tiled inference on a large image:

```bash
python scripts/sliced_predict.py \
  --source data/test/ocean_scene_10000x10000.png \
  --weights runs/YOLO11n/yolo11n_fold1_img800_base_b4w2/weights/best_0.92475.pt \
  --tile-size 800 \
  --overlap 0.20
```

The command measures model loading, slicing, inference, global class-wise NMS,
rendering, and saving. It writes the rendered PNG, merged predictions, and timing
summary to `runs/sliced_predict/` by default. Use `--model-type rtdetr` for an
RT-DETR checkpoint when its path does not contain `rtdetr` or `rt-detr`.
