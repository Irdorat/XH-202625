# Per-class checkpoint metrics

Validation metrics for all 48 curated checkpoints and all 25 target classes (1200 checkpoint-class rows).

## Files

- [per_class.csv](per_class.csv): consolidated checkpoint-class rows.
- [checkpoints.csv](checkpoints.csv): aggregate rerun metrics and signed deltas from embedded selection metrics.
- [by_checkpoint/](by_checkpoint/): one 25-row CSV per checkpoint.
- [manifest.json](manifest.json): protocol, environment, dataset fingerprints, and artifact counts.

## Protocol

- Every checkpoint uses the validation manifest for its own label version and fold.
- Standard Ultralytics detection validation: image size 800, NMS IoU 0.7, max detections 300, FP32, no test-time augmentation.
- Precision, Recall, and F1 use the single global confidence threshold maximizing the smoothed mean F1 curve.
- FDR equals 1 minus Precision.
- mAP50 is AP at IoU 0.50; mAP50-95 averages AP over IoU 0.50:0.05:0.95.
- Images and Instances count validation images containing the class and its ground-truth objects, respectively.
- Values are per-fold validation results, not test metrics or cross-validation averages.

Aggregate values are fresh reruns. Maximum absolute deltas from embedded selection metrics: Precision 0.05771, Recall 0.03109, mAP50 0.00136, and mAP50-95 0.00228.

Precision and Recall are recomputed at the global max-F1 confidence operating point and can move more than AP; checkpoints.csv preserves both sets of values and their signed deltas.

## Reproduction

    python scripts/evaluate.py --original-data-dir /path/to/original-label-data --new-data-dir /path/to/new-label-data --work-dir runs/checkpoint_per_class_metrics --image-size 800 --batch-size 64 --workers 16 --device 0 --framework-revision SOURCE_ID

Environment used for the committed report:

- Python 3.10.12
- PyTorch 2.13.0+cu130 with CUDA 13.0
- Ultralytics 8.4.7
- GPU NVIDIA GeForce RTX 5090
- YOLO source identifier source-tree-sha256:a2e714bafc0b84c3064c9b9384938c5adf0c90ade9ea9f7d982191b1a4986517

## Checkpoint index

| Labels | Model | Fold | Checkpoint | Per-class CSV | Precision | Recall | FDR | mAP50 | mAP50-95 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| new_labels | yolo26n | 0 | [map_best.pt](../new_labels/yolo26n/fold0/weights/map_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold0/map_best.csv) | 0.93174 | 0.87197 | 0.06826 | 0.90950 | 0.73934 |
| new_labels | yolo26n | 0 | [precision_fdr_best.pt](../new_labels/yolo26n/fold0/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold0/precision_fdr_best.csv) | 0.95034 | 0.86714 | 0.04966 | 0.90667 | 0.73580 |
| new_labels | yolo26n | 0 | [recall_best.pt](../new_labels/yolo26n/fold0/weights/recall_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold0/recall_best.csv) | 0.88494 | 0.89244 | 0.11506 | 0.90416 | 0.73189 |
| new_labels | yolo26s | 0 | [map_best.pt](../new_labels/yolo26s/fold0/weights/map_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold0/map_best.csv) | 0.89507 | 0.90264 | 0.10493 | 0.92280 | 0.75510 |
| new_labels | yolo26s | 0 | [precision_fdr_best.pt](../new_labels/yolo26s/fold0/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold0/precision_fdr_best.csv) | 0.96097 | 0.87758 | 0.03903 | 0.91899 | 0.74972 |
| new_labels | yolo26s | 0 | [recall_best.pt](../new_labels/yolo26s/fold0/weights/recall_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold0/recall_best.csv) | 0.88762 | 0.91269 | 0.11238 | 0.92519 | 0.73922 |
| new_labels | yolo26n | 1 | [map_best.pt](../new_labels/yolo26n/fold1/weights/map_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold1/map_best.csv) | 0.90720 | 0.88100 | 0.09280 | 0.91314 | 0.74766 |
| new_labels | yolo26n | 1 | [precision_fdr_best.pt](../new_labels/yolo26n/fold1/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold1/precision_fdr_best.csv) | 0.93704 | 0.86459 | 0.06296 | 0.91571 | 0.74697 |
| new_labels | yolo26n | 1 | [recall_best.pt](../new_labels/yolo26n/fold1/weights/recall_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold1/recall_best.csv) | 0.87966 | 0.89038 | 0.12034 | 0.91585 | 0.72075 |
| new_labels | yolo26s | 1 | [map_best.pt](../new_labels/yolo26s/fold1/weights/map_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold1/map_best.csv) | 0.91921 | 0.89341 | 0.08079 | 0.92529 | 0.76332 |
| new_labels | yolo26s | 1 | [precision_fdr_best.pt](../new_labels/yolo26s/fold1/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold1/precision_fdr_best.csv) | 0.93859 | 0.84669 | 0.06141 | 0.90928 | 0.74048 |
| new_labels | yolo26s | 1 | [recall_best.pt](../new_labels/yolo26s/fold1/weights/recall_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold1/recall_best.csv) | 0.88901 | 0.90673 | 0.11099 | 0.92182 | 0.75816 |
| new_labels | yolo26n | 2 | [map_best.pt](../new_labels/yolo26n/fold2/weights/map_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold2/map_best.csv) | 0.91607 | 0.89828 | 0.08393 | 0.94380 | 0.76936 |
| new_labels | yolo26n | 2 | [precision_fdr_best.pt](../new_labels/yolo26n/fold2/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold2/precision_fdr_best.csv) | 0.94820 | 0.87591 | 0.05180 | 0.92716 | 0.75554 |
| new_labels | yolo26n | 2 | [recall_best.pt](../new_labels/yolo26n/fold2/weights/recall_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold2/recall_best.csv) | 0.89978 | 0.91579 | 0.10022 | 0.92643 | 0.74512 |
| new_labels | yolo26s | 2 | [map_best.pt](../new_labels/yolo26s/fold2/weights/map_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold2/map_best.csv) | 0.93564 | 0.89749 | 0.06436 | 0.95121 | 0.78294 |
| new_labels | yolo26s | 2 | [precision_fdr_best.pt](../new_labels/yolo26s/fold2/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold2/precision_fdr_best.csv) | 0.96242 | 0.88783 | 0.03758 | 0.93660 | 0.77345 |
| new_labels | yolo26s | 2 | [recall_best.pt](../new_labels/yolo26s/fold2/weights/recall_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold2/recall_best.csv) | 0.89513 | 0.92972 | 0.10487 | 0.95332 | 0.76261 |
| new_labels | yolo26n | 3 | [map_best.pt](../new_labels/yolo26n/fold3/weights/map_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold3/map_best.csv) | 0.91491 | 0.89013 | 0.08509 | 0.92331 | 0.75319 |
| new_labels | yolo26n | 3 | [precision_fdr_best.pt](../new_labels/yolo26n/fold3/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold3/precision_fdr_best.csv) | 0.93960 | 0.86077 | 0.06040 | 0.90611 | 0.74653 |
| new_labels | yolo26n | 3 | [recall_best.pt](../new_labels/yolo26n/fold3/weights/recall_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold3/recall_best.csv) | 0.91575 | 0.87801 | 0.08425 | 0.91640 | 0.74414 |
| new_labels | yolo26s | 3 | [map_best.pt](../new_labels/yolo26s/fold3/weights/map_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold3/map_best.csv) | 0.93702 | 0.87805 | 0.06298 | 0.92593 | 0.77277 |
| new_labels | yolo26s | 3 | [precision_fdr_best.pt](../new_labels/yolo26s/fold3/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold3/precision_fdr_best.csv) | 0.95444 | 0.86808 | 0.04556 | 0.91082 | 0.76142 |
| new_labels | yolo26s | 3 | [recall_best.pt](../new_labels/yolo26s/fold3/weights/recall_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold3/recall_best.csv) | 0.87812 | 0.89256 | 0.12188 | 0.92019 | 0.73248 |
| new_labels | yolo26n | 4 | [map_best.pt](../new_labels/yolo26n/fold4/weights/map_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold4/map_best.csv) | 0.91517 | 0.91970 | 0.08483 | 0.95136 | 0.77496 |
| new_labels | yolo26n | 4 | [precision_fdr_best.pt](../new_labels/yolo26n/fold4/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold4/precision_fdr_best.csv) | 0.91975 | 0.92116 | 0.08025 | 0.94377 | 0.76833 |
| new_labels | yolo26n | 4 | [recall_best.pt](../new_labels/yolo26n/fold4/weights/recall_best.pt) | [CSV](by_checkpoint/new_labels/yolo26n/fold4/recall_best.csv) | 0.89535 | 0.94504 | 0.10465 | 0.94761 | 0.77052 |
| new_labels | yolo26s | 4 | [map_best.pt](../new_labels/yolo26s/fold4/weights/map_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold4/map_best.csv) | 0.94698 | 0.91873 | 0.05302 | 0.95539 | 0.78297 |
| new_labels | yolo26s | 4 | [precision_fdr_best.pt](../new_labels/yolo26s/fold4/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold4/precision_fdr_best.csv) | 0.96965 | 0.91605 | 0.03035 | 0.95277 | 0.77810 |
| new_labels | yolo26s | 4 | [recall_best.pt](../new_labels/yolo26s/fold4/weights/recall_best.pt) | [CSV](by_checkpoint/new_labels/yolo26s/fold4/recall_best.csv) | 0.93332 | 0.94344 | 0.06668 | 0.95947 | 0.77886 |
| original_labels | yolo26n | 0 | [map_best.pt](../original_labels/yolo26n/fold0/weights/map_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold0/map_best.csv) | 0.91926 | 0.88406 | 0.08074 | 0.91803 | 0.74550 |
| original_labels | yolo26n | 0 | [precision_fdr_best.pt](../original_labels/yolo26n/fold0/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold0/precision_fdr_best.csv) | 0.95809 | 0.87434 | 0.04191 | 0.91690 | 0.74115 |
| original_labels | yolo26n | 0 | [recall_best.pt](../original_labels/yolo26n/fold0/weights/recall_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold0/recall_best.csv) | 0.89651 | 0.90321 | 0.10349 | 0.91565 | 0.73624 |
| original_labels | yolo26n | 1 | [map_best.pt](../original_labels/yolo26n/fold1/weights/map_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold1/map_best.csv) | 0.91372 | 0.90625 | 0.08628 | 0.93054 | 0.75930 |
| original_labels | yolo26n | 1 | [precision_fdr_best.pt](../original_labels/yolo26n/fold1/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold1/precision_fdr_best.csv) | 0.94004 | 0.89378 | 0.05996 | 0.92696 | 0.75686 |
| original_labels | yolo26n | 1 | [recall_best.pt](../original_labels/yolo26n/fold1/weights/recall_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold1/recall_best.csv) | 0.88051 | 0.92203 | 0.11949 | 0.92187 | 0.75141 |
| original_labels | yolo26n | 2 | [map_best.pt](../original_labels/yolo26n/fold2/weights/map_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold2/map_best.csv) | 0.91165 | 0.93108 | 0.08835 | 0.94684 | 0.76253 |
| original_labels | yolo26n | 2 | [precision_fdr_best.pt](../original_labels/yolo26n/fold2/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold2/precision_fdr_best.csv) | 0.94920 | 0.90259 | 0.05080 | 0.93365 | 0.75219 |
| original_labels | yolo26n | 2 | [recall_best.pt](../original_labels/yolo26n/fold2/weights/recall_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold2/recall_best.csv) | 0.91165 | 0.93108 | 0.08835 | 0.94684 | 0.76253 |
| original_labels | yolo26n | 3 | [map_best.pt](../original_labels/yolo26n/fold3/weights/map_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold3/map_best.csv) | 0.90638 | 0.90075 | 0.09362 | 0.92912 | 0.75484 |
| original_labels | yolo26n | 3 | [precision_fdr_best.pt](../original_labels/yolo26n/fold3/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold3/precision_fdr_best.csv) | 0.94233 | 0.86498 | 0.05767 | 0.91478 | 0.74755 |
| original_labels | yolo26n | 3 | [recall_best.pt](../original_labels/yolo26n/fold3/weights/recall_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold3/recall_best.csv) | 0.88257 | 0.91100 | 0.11743 | 0.93191 | 0.74010 |
| original_labels | yolo26n | 4 | [map_best.pt](../original_labels/yolo26n/fold4/weights/map_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold4/map_best.csv) | 0.95576 | 0.93804 | 0.04424 | 0.95582 | 0.77780 |
| original_labels | yolo26n | 4 | [precision_fdr_best.pt](../original_labels/yolo26n/fold4/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold4/precision_fdr_best.csv) | 0.96907 | 0.92542 | 0.03093 | 0.95550 | 0.77615 |
| original_labels | yolo26n | 4 | [recall_best.pt](../original_labels/yolo26n/fold4/weights/recall_best.pt) | [CSV](by_checkpoint/original_labels/yolo26n/fold4/recall_best.csv) | 0.93255 | 0.94297 | 0.06745 | 0.95143 | 0.77357 |
| original_labels | yolo26s | 4 | [map_best.pt](../original_labels/yolo26s/fold4/weights/map_best.pt) | [CSV](by_checkpoint/original_labels/yolo26s/fold4/map_best.csv) | 0.95476 | 0.93844 | 0.04524 | 0.96159 | 0.78618 |
| original_labels | yolo26s | 4 | [precision_fdr_best.pt](../original_labels/yolo26s/fold4/weights/precision_fdr_best.pt) | [CSV](by_checkpoint/original_labels/yolo26s/fold4/precision_fdr_best.csv) | 0.96746 | 0.93772 | 0.03254 | 0.95822 | 0.77739 |
| original_labels | yolo26s | 4 | [recall_best.pt](../original_labels/yolo26s/fold4/weights/recall_best.pt) | [CSV](by_checkpoint/original_labels/yolo26s/fold4/recall_best.csv) | 0.95672 | 0.94303 | 0.04328 | 0.95857 | 0.77764 |
