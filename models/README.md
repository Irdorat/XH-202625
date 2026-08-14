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
└── new_labels/
    ├── yolo26n/fold0..4/weights/
    └── yolo26s/fold0..4/weights/
```

Each fold contains `map_best.pt` and `precision_fdr_best.pt`. These selected
model binaries are committed through Git LFS. Run `git lfs install` before
cloning or pulling the repository. `trained/SHA256SUMS` records the checksum of
every tracked checkpoint.

Do not commit unselected epoch checkpoints, pretrained weights, run directories,
or model archives. Include the model name, experiment name, expected filename,
