# Per-class checkpoint metrics

Selection-time final validation metrics for 10 checkpoints and all 25 target
classes (250 checkpoint-class rows).

## Files

- `per_class.csv`: consolidated checkpoint-class rows.
- `checkpoints.csv`: aggregate selection-time metrics.
- `by_checkpoint/`: one 25-row CSV per checkpoint.
- `manifest.json`: protocol, provenance, and precision limits.

## Protocol

- Each checkpoint uses its final-dataset validation split for the same fold.
- Ultralytics 8.4.126 detection validation at image size 800.
- Precision, Recall, FDR, F1, mAP50, and mAP50-95 follow the existing branch schema.
- FDR is `1 - Precision` and F1 is the harmonic mean of Precision and Recall.
- These values were parsed from the final `best.pt` validation tables produced
  at training completion; no new validation was run while another authorized
  FSC training job occupied the GPU.
- The source log prints per-class metrics to three decimal places. Values are
  padded to eight decimal places for CSV schema compatibility, not to claim
  additional numerical precision. Aggregate metrics retain their five-decimal
  training-completion values.
