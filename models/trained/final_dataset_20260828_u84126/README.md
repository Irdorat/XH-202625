# Final dataset YOLO26 results (2026-08-28)

This directory is an additive result set. It does not replace the existing
`original_labels/` or `new_labels/` collections.

## Protocol

- Dataset pool: 9666 images, 25 classes
- Existing fold0-fold4 validation membership; every other final-dataset image is training data
- Image size: 800
- Optimizer: MuSGD (`lr0=0.005`)
- Maximum epochs: 150; early-stopping patience: 50
- YOLO26n batch: 64; YOLO26s batch: 32
- Ultralytics: 8.4.126 from `/root/miniconda3/lib/python3.10/site-packages/ultralytics`
- Selection: each run's Ultralytics `best.pt`

## Five-fold aggregate

| Model | Precision | Recall | mAP50 | mAP50-95 | mAP50-95 std |
|---|---:|---:|---:|---:|---:|
| yolo26n | 0.95435 | 0.92479 | 0.94788 | 0.86018 | 0.01021 |
| yolo26s | 0.95221 | 0.93784 | 0.95806 | 0.87468 | 0.01190 |

## Fold results

| Fold | YOLO26n mAP50-95 | YOLO26s mAP50-95 | S minus N |
|---:|---:|---:|---:|
| 0 | 0.86294 | 0.88067 | +0.01773 |
| 1 | 0.84408 | 0.85422 | +0.01014 |
| 2 | 0.86439 | 0.87408 | +0.00969 |
| 3 | 0.85486 | 0.87384 | +0.01898 |
| 4 | 0.87462 | 0.89057 | +0.01595 |

## Fold4 MS and FSC

| Class | Model | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| MS | yolo26n | 238 | 447 | 0.849 | 0.843 | 0.911 | 0.720 |
| MS | yolo26s | 238 | 447 | 0.859 | 0.858 | 0.918 | 0.744 |
| FSC | yolo26n | 13 | 93 | 0.916 | 0.495 | 0.659 | 0.387 |
| FSC | yolo26s | 13 | 93 | 0.982 | 0.596 | 0.708 | 0.446 |

YOLO26s outperformed YOLO26n on mAP50-95 in every matched fold. FSC
remains the principal long-tail weakness: fold4 contains 93 FSC instances
in only 13 validation images, and FSC objects are substantially smaller than MS.

## Contents

- `aggregate_metrics.csv`: one row for each of the 10 completed runs
- `per_class_metrics.csv`: 250 rows (10 runs x 25 classes)
- `fold4_ms_fsc_metrics.csv`: focused MS/FSC comparison
- `dataset_splits.csv`: train/validation image and box counts
- `manifest.json` and `SHA256SUMS`: provenance and file integrity
- `<model>/fold<k>/weights/best.pt`: selected checkpoint tracked by Git LFS
- Each fold also contains `args.yaml`, `results.csv`, result curves, and confusion matrices

## Comparability note

Validation membership follows the prior fold basename lists, but the final dataset
contains revised annotations and a small number of revised image files. Metrics here
must not be treated as a strict same-annotation comparison with older result sets.
