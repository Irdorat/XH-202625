# Scripts

This directory contains command-line entry points for project workflows.

```text
scripts/
├── prepare_data.py    # Build and validate split manifests
├── train.py           # Train a configured experiment
├── evaluate.py        # Evaluate a checkpoint on a fixed split
└── predict.py         # Run inference on images
```

Scripts should remain thin: parse arguments, load configuration, call reusable code from `src/ml_comp/`, and write outputs to an explicit run directory.

An entry point is not considered implemented while its file is empty. Commands and defaults must use repository-relative paths and must not modify `data/raw`.
