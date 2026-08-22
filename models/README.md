# Models

This directory separates external pretrained weights from checkpoints selected by the team.

```text
models/
├── pretrained/    # Original pretrained checkpoints
└── trained/       # Selected project checkpoints tracked with Git LFS
```

The selected baseline v1 checkpoint is `trained/baseline_v1_best.pt`.

The current multi-fold checkpoint collection uses this layout:

```text
trained/
├── original_labels/
│   ├── yolo26n/fold0..4/weights/
│   └── yolo26s/fold4/weights/
├── new_labels/
│   ├── yolo26n/fold0..4/weights/
│   └── yolo26s/fold0..4/weights/
├── metrics/        # Aggregate and 25-class validation reports
└── SHA256SUMS
```

Each fold contains `map_best.pt`, `precision_fdr_best.pt`, and
`recall_best.pt`. These selected model binaries are committed through Git LFS.
Run `git lfs install` before cloning or pulling the repository.
`trained/SHA256SUMS` records the checksum of every tracked checkpoint.

## Per-class metrics

Fresh standard-validation reruns for all 48 curated checkpoints are tracked in
[`trained/metrics/`](trained/metrics/README.md). Each checkpoint has one
25-row CSV with Images, Instances, Precision, Recall, FDR, F1, mAP50, and
mAP50-95 for every target class.

- [`trained/metrics/per_class.csv`](trained/metrics/per_class.csv) contains all
  1,200 checkpoint-class rows.
- [`trained/metrics/checkpoints.csv`](trained/metrics/checkpoints.csv) contains
  aggregate rerun metrics and deltas from the embedded selection metrics.
- [`trained/metrics/manifest.json`](trained/metrics/manifest.json) records the
  evaluation protocol, runtime environment, and validation-data fingerprints.

## Checkpoint selection

- `map_best.pt` is the checkpoint selected by the highest validation
  `metrics/mAP50-95(B)`.
- `precision_fdr_best.pt` is the checkpoint selected by the highest validation
  `metrics/precision(B)`, equivalently the lowest FDR under the project
  definition `FDR = 1 - Precision`.
- `recall_best.pt` is the checkpoint selected by the highest validation
  `metrics/recall(B)`.

## Validation metrics

The tables below report aggregate bounding-box metrics on the validation split
of the corresponding fold. Values come from the selection-time validation
metrics embedded in each uploaded checkpoint and are rounded to five decimal
places. They are per-fold validation results, not test-set results or
cross-validation averages. Results for `original_labels` and `new_labels` use
their respective annotation versions and should be compared with that
difference in mind. For fresh reruns and per-class values, use the reports in
[`trained/metrics/`](trained/metrics/README.md).

### Original labels — YOLO26n

| Fold | Weight | Selected epoch | Precision | Recall | FDR | mAP50 | mAP50–95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | [`map_best.pt`](trained/original_labels/yolo26n/fold0/weights/map_best.pt) | 113 | 0.92039 | 0.88229 | 0.07961 | 0.91862 | 0.74666 |
| 0 | [`precision_fdr_best.pt`](trained/original_labels/yolo26n/fold0/weights/precision_fdr_best.pt) | 121 | 0.95559 | 0.87576 | 0.04441 | 0.91683 | 0.74185 |
| 0 | [`recall_best.pt`](trained/original_labels/yolo26n/fold0/weights/recall_best.pt) | 84 | 0.89816 | 0.90425 | 0.10184 | 0.91588 | 0.73632 |
| 1 | [`map_best.pt`](trained/original_labels/yolo26n/fold1/weights/map_best.pt) | 114 | 0.91477 | 0.90451 | 0.08523 | 0.93067 | 0.75881 |
| 1 | [`precision_fdr_best.pt`](trained/original_labels/yolo26n/fold1/weights/precision_fdr_best.pt) | 136 | 0.94339 | 0.89055 | 0.05661 | 0.92677 | 0.75659 |
| 1 | [`recall_best.pt`](trained/original_labels/yolo26n/fold1/weights/recall_best.pt) | 116 | 0.87968 | 0.92187 | 0.12032 | 0.92215 | 0.75065 |
| 2 | [`map_best.pt`](trained/original_labels/yolo26n/fold2/weights/map_best.pt) | 77 | 0.91143 | 0.93073 | 0.08857 | 0.94618 | 0.76127 |
| 2 | [`precision_fdr_best.pt`](trained/original_labels/yolo26n/fold2/weights/precision_fdr_best.pt) | 92 | 0.95098 | 0.90012 | 0.04902 | 0.93370 | 0.75188 |
| 2 | [`recall_best.pt`](trained/original_labels/yolo26n/fold2/weights/recall_best.pt) | 77 | 0.91143 | 0.93073 | 0.08857 | 0.94618 | 0.76127 |
| 3 | [`map_best.pt`](trained/original_labels/yolo26n/fold3/weights/map_best.pt) | 86 | 0.90398 | 0.90076 | 0.09602 | 0.92891 | 0.75415 |
| 3 | [`precision_fdr_best.pt`](trained/original_labels/yolo26n/fold3/weights/precision_fdr_best.pt) | 130 | 0.94400 | 0.86474 | 0.05600 | 0.91484 | 0.74710 |
| 3 | [`recall_best.pt`](trained/original_labels/yolo26n/fold3/weights/recall_best.pt) | 56 | 0.88141 | 0.91185 | 0.11859 | 0.93152 | 0.73901 |
| 4 | [`map_best.pt`](trained/original_labels/yolo26n/fold4/weights/map_best.pt) | 115 | 0.95644 | 0.93778 | 0.04356 | 0.95591 | 0.77856 |
| 4 | [`precision_fdr_best.pt`](trained/original_labels/yolo26n/fold4/weights/precision_fdr_best.pt) | 128 | 0.96912 | 0.92509 | 0.03088 | 0.95573 | 0.77595 |
| 4 | [`recall_best.pt`](trained/original_labels/yolo26n/fold4/weights/recall_best.pt) | 144 | 0.93056 | 0.94325 | 0.06944 | 0.95155 | 0.77296 |

### Original labels — YOLO26s

| Fold | Weight | Selected epoch | Precision | Recall | FDR | mAP50 | mAP50–95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | [`map_best.pt`](trained/original_labels/yolo26s/fold4/weights/map_best.pt) | 109 | 0.95607 | 0.93714 | 0.04393 | 0.96160 | 0.78669 |
| 4 | [`precision_fdr_best.pt`](trained/original_labels/yolo26s/fold4/weights/precision_fdr_best.pt) | 138 | 0.96733 | 0.93768 | 0.03267 | 0.95807 | 0.77700 |
| 4 | [`recall_best.pt`](trained/original_labels/yolo26s/fold4/weights/recall_best.pt) | 137 | 0.95655 | 0.94309 | 0.04345 | 0.95842 | 0.77746 |

### New labels — YOLO26n

| Fold | Weight | Selected epoch | Precision | Recall | FDR | mAP50 | mAP50–95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | [`map_best.pt`](trained/new_labels/yolo26n/fold0/weights/map_best.pt) | 105 | 0.93209 | 0.87158 | 0.06791 | 0.90943 | 0.73905 |
| 0 | [`precision_fdr_best.pt`](trained/new_labels/yolo26n/fold0/weights/precision_fdr_best.pt) | 138 | 0.95094 | 0.86745 | 0.04906 | 0.90667 | 0.73596 |
| 0 | [`recall_best.pt`](trained/new_labels/yolo26n/fold0/weights/recall_best.pt) | 109 | 0.88495 | 0.89290 | 0.11505 | 0.90434 | 0.73170 |
| 1 | [`map_best.pt`](trained/new_labels/yolo26n/fold1/weights/map_best.pt) | 146 | 0.90734 | 0.88138 | 0.09266 | 0.91403 | 0.74847 |
| 1 | [`precision_fdr_best.pt`](trained/new_labels/yolo26n/fold1/weights/precision_fdr_best.pt) | 134 | 0.93529 | 0.86429 | 0.06471 | 0.91642 | 0.74701 |
| 1 | [`recall_best.pt`](trained/new_labels/yolo26n/fold1/weights/recall_best.pt) | 55 | 0.87996 | 0.88993 | 0.12004 | 0.91577 | 0.72104 |
| 2 | [`map_best.pt`](trained/new_labels/yolo26n/fold2/weights/map_best.pt) | 103 | 0.91616 | 0.89861 | 0.08384 | 0.94381 | 0.76923 |
| 2 | [`precision_fdr_best.pt`](trained/new_labels/yolo26n/fold2/weights/precision_fdr_best.pt) | 123 | 0.94872 | 0.87532 | 0.05128 | 0.92708 | 0.75521 |
| 2 | [`recall_best.pt`](trained/new_labels/yolo26n/fold2/weights/recall_best.pt) | 77 | 0.90158 | 0.91549 | 0.09842 | 0.92649 | 0.74567 |
| 3 | [`map_best.pt`](trained/new_labels/yolo26n/fold3/weights/map_best.pt) | 76 | 0.91453 | 0.88862 | 0.08547 | 0.92434 | 0.75342 |
| 3 | [`precision_fdr_best.pt`](trained/new_labels/yolo26n/fold3/weights/precision_fdr_best.pt) | 108 | 0.94152 | 0.86035 | 0.05848 | 0.90627 | 0.74737 |
| 3 | [`recall_best.pt`](trained/new_labels/yolo26n/fold3/weights/recall_best.pt) | 63 | 0.88842 | 0.89727 | 0.11158 | 0.91752 | 0.74642 |
| 4 | [`map_best.pt`](trained/new_labels/yolo26n/fold4/weights/map_best.pt) | 84 | 0.90412 | 0.92723 | 0.09588 | 0.95129 | 0.77546 |
| 4 | [`precision_fdr_best.pt`](trained/new_labels/yolo26n/fold4/weights/precision_fdr_best.pt) | 103 | 0.95968 | 0.89007 | 0.04032 | 0.94241 | 0.76790 |
| 4 | [`recall_best.pt`](trained/new_labels/yolo26n/fold4/weights/recall_best.pt) | 112 | 0.89635 | 0.94504 | 0.10365 | 0.94773 | 0.77068 |

### New labels — YOLO26s

| Fold | Weight | Selected epoch | Precision | Recall | FDR | mAP50 | mAP50–95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | [`map_best.pt`](trained/new_labels/yolo26s/fold0/weights/map_best.pt) | 147 | 0.95278 | 0.87943 | 0.04722 | 0.92283 | 0.75524 |
| 0 | [`precision_fdr_best.pt`](trained/new_labels/yolo26s/fold0/weights/precision_fdr_best.pt) | 123 | 0.96208 | 0.87759 | 0.03792 | 0.91900 | 0.74995 |
| 0 | [`recall_best.pt`](trained/new_labels/yolo26s/fold0/weights/recall_best.pt) | 52 | 0.88700 | 0.91260 | 0.11300 | 0.92517 | 0.73956 |
| 1 | [`map_best.pt`](trained/new_labels/yolo26s/fold1/weights/map_best.pt) | 100 | 0.91889 | 0.89412 | 0.08111 | 0.92571 | 0.76337 |
| 1 | [`precision_fdr_best.pt`](trained/new_labels/yolo26s/fold1/weights/precision_fdr_best.pt) | 48 | 0.93895 | 0.84623 | 0.06105 | 0.90934 | 0.74112 |
| 1 | [`recall_best.pt`](trained/new_labels/yolo26s/fold1/weights/recall_best.pt) | 88 | 0.88973 | 0.90714 | 0.11027 | 0.92173 | 0.75830 |
| 2 | [`map_best.pt`](trained/new_labels/yolo26s/fold2/weights/map_best.pt) | 111 | 0.92759 | 0.90556 | 0.07241 | 0.95128 | 0.78371 |
| 2 | [`precision_fdr_best.pt`](trained/new_labels/yolo26s/fold2/weights/precision_fdr_best.pt) | 142 | 0.96226 | 0.88745 | 0.03774 | 0.93653 | 0.77409 |
| 2 | [`recall_best.pt`](trained/new_labels/yolo26s/fold2/weights/recall_best.pt) | 28 | 0.89505 | 0.92989 | 0.10495 | 0.95334 | 0.76267 |
| 3 | [`map_best.pt`](trained/new_labels/yolo26s/fold3/weights/map_best.pt) | 130 | 0.93977 | 0.87649 | 0.06023 | 0.92564 | 0.77210 |
| 3 | [`precision_fdr_best.pt`](trained/new_labels/yolo26s/fold3/weights/precision_fdr_best.pt) | 108 | 0.95452 | 0.86796 | 0.04548 | 0.91084 | 0.76161 |
| 3 | [`recall_best.pt`](trained/new_labels/yolo26s/fold3/weights/recall_best.pt) | 28 | 0.87808 | 0.89206 | 0.12192 | 0.92019 | 0.73293 |
| 4 | [`map_best.pt`](trained/new_labels/yolo26s/fold4/weights/map_best.pt) | 73 | 0.94658 | 0.91878 | 0.05342 | 0.95523 | 0.78304 |
| 4 | [`precision_fdr_best.pt`](trained/new_labels/yolo26s/fold4/weights/precision_fdr_best.pt) | 118 | 0.96978 | 0.91558 | 0.03022 | 0.95334 | 0.77833 |
| 4 | [`recall_best.pt`](trained/new_labels/yolo26s/fold4/weights/recall_best.pt) | 69 | 0.93286 | 0.94336 | 0.06714 | 0.95963 | 0.77911 |

Do not commit unselected epoch checkpoints, pretrained weights, run directories,
or model archives. Include the model name, experiment name, expected filename,
checksum, and corresponding configuration in the handoff notes.
