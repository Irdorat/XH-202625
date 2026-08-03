# Dataset Splits

This directory stores reproducible text manifests rather than copies of images.

```text
data/splits/
├── baseline_v1/    # Reconstructed historical split: 3,585 train / 896 val
└── baseline_v2/    # Separate scene-aware split for baseline v2
```

Every split directory should contain `train.txt` and `val.txt`. Each line points to one image using a repository-relative path.

Rules:

- never overwrite a finalized split;
- create a new version when split logic changes;
- keep all images originating from the same scene in one subset;
- verify that rare classes remain represented in training;
- document the seed, grouping rule, image counts, object counts, and class distribution.

Model improvements may be compared only when evaluated on the same split with the same protocol.
