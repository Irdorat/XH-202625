# Baseline v1: Results and Current Status

## Purpose

Baseline v1 demonstrates the performance of an unmodified RT-DETR-L architecture on the provided dataset. It is preserved as the initial project reference and will not be retrained during baseline v2 development.

## Model and Training Configuration

| Parameter | Value |
|---|---:|
| Model | RT-DETR-L |
| Pretraining | COCO |
| Input size | 640×640 |
| Epochs | 50 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Training fraction used by the historical run | 0.5 |

Selected checkpoint: `models/trained/baseline_v1_best.pt`.

## Validation Dataset

The original baseline v1 split was reconstructed from the historical split procedure and validation class fingerprint. It is now fixed in:

```text
data/splits/baseline_v1/
├── train.txt       # 3,585 images
└── val.txt         # 896 images
```

The raw validation annotations contain 4,295 entries. One exact duplicated FSC annotation is removed by Ultralytics and by the custom evaluator, leaving 4,294 ground-truth objects. All 896 images have annotation files, and no corrupt or background-only images were reported during validation.

This split reproduces the validation subset used during baseline v1 training. It should remain unchanged.

## Evaluation Protocol

The full validation evaluation is implemented in `notebooks/baseline_v1/baseline_v1_results.ipynb`.

Competition-oriented detection counts use:

- a fixed confidence threshold of `0.50`;
- class-aware NMS with IoU threshold `0.50`;
- class-aware one-to-one matching with IoU threshold `0.50`;
- micro-aggregation of TP, FP, and FN over the complete validation set.

The provisional metrics are defined as:

```text
Recall   = TP / (TP + FN)
FAR-pred = FP / (TP + FP)
FAR-gt   = FP / (TP + FN)
```

The official competition FAR formula is not defined in the available documents. Both FAR variants must therefore be treated as provisional.

Standard Precision, Recall, mAP50, and mAP50–95 are also calculated separately by the Ultralytics validator. These values must not be mixed with the fixed-threshold custom operating-point metrics.

## Full Validation Results

### Selected operating point

Results at confidence `0.50` after class-aware NMS:

| Metric | Result | Current target | Status |
|---|---:|---:|---|
| Ground-truth objects | 4,294 | — | — |
| Predictions | 4,706 | — | — |
| TP / FP / FN | 3,879 / 827 / 415 | — | — |
| Precision | 0.824 | — | — |
| Recall | 0.903 | ≥0.85 | passes |
| F1 score | 0.862 | — | — |
| FAR-pred | 0.176 | ≤0.20* | passes provisionally |
| FAR-gt | 0.193 | ≤0.20* | passes provisionally |

\* The FAR targets are evaluated using the two provisional team definitions.

Without the additional class-aware NMS, the same confidence threshold gives Precision `0.807`, Recall `0.906`, FAR-pred `0.193`, and FAR-gt `0.217`. NMS therefore reduces false detections enough for both provisional FAR variants to fall below 0.20, with a small Recall reduction.

### Standard Ultralytics validation

| Metric | Result |
|---|---:|
| Precision | 0.774 |
| Recall | 0.795 |
| mAP50 | 0.801 |
| mAP50–95 | 0.449 |

The custom Recall of `0.903` and Ultralytics Recall of `0.795` answer different questions. The first describes the explicitly selected fixed-confidence competition operating point under the custom matching procedure. The second is the standard Ultralytics aggregate metric. Future comparisons must report both using the same protocol.

## Class-Imbalance Findings

Aggregate performance hides severe failures on the rarest classes:

| Class | Validation instances | Ultralytics Recall | AP50 |
|---|---:|---:|---:|
| HM | 3 | 0.000 | 0.100 |
| LQS | 5 | 0.000 | 0.062 |
| FSC | 83 | 0.000 | 0.000 |
| A18_KC-10 | 42 | 0.288 | 0.789 |

Several aircraft classes already have AP50 above 0.95, while HM, LQS, and FSC remain effectively undetected. This imbalance motivates Fair Training, Copy-Paste, and Scene Mixing in baseline v2.

## Processing Time

The first full-validation inference pass was measured on an NVIDIA GeForce RTX 3060 Laptop GPU using batches of eight provided image crops:

| Stage | Mean time per image |
|---|---:|
| Preprocessing | 1.75 ms |
| Model inference | 21.89 ms |
| Postprocessing | 0.64 ms |
| Model stages total | 24.28 ms |
| Observed wall time | 36.71 ms |

These measurements describe the provided crop-sized images and the current notebook pipeline. They do not establish end-to-end processing time for a complete 10,000×10,000-pixel scene.

## Competition Requirements

`docs/general/project_context.pdf` specifies three performance conditions for evaluation on the closed test set:

| Indicator | Requirement | Baseline v1 evidence |
|---|---:|---|
| Overall Recall | ≥0.85 | 0.903 under the documented custom operating-point protocol |
| FAR | ≤0.20 | 0.176 and 0.193 under two provisional definitions |
| Processing time | ≤20 seconds per 10,000×10,000 image | not yet measured |

Runtime is expected to be evaluated on one NVIDIA RTX 3090 or an equivalent domestic GPU/NPU. Later information received by the team indicates that ranking may use a weighted combination of the three indicators. The exact formula, weights, normalization, and official FAR definition are not available in the current documents.

Baseline v1 therefore satisfies the current local Recall target and both provisional FAR targets on the reconstructed validation set. Full compliance with the competition requirements is not yet demonstrated because the official FAR definition and the end-to-end 10k×10k runtime are unavailable.

## Historical Metrics

Earlier saved baseline results reported Precision `0.849`, Recall `0.761`, mAP50 `0.808`, and mAP50–95 `0.453`. These values came from a separate historical validation state and are retained only as experiment history. The controlled 896-image validation reported above is now the baseline v1 reference evaluation.

## Remaining Competition-Oriented Work

1. Obtain or reproduce the official FAR definition and evaluator.
2. Implement the complete 10k×10k pipeline: slicing, batched inference, coordinate merging, and global suppression.
3. Benchmark end-to-end runtime on an RTX 3090 or equivalent device.
4. Preserve the baseline v1 split and evaluation protocol for regression testing.
5. Evaluate baseline v2 ablations on a separately versioned, scene-aware split with a corresponding control run.
