# Source Package

`src/ml_comp/` contains reusable project logic shared by scripts, notebooks, and the future Docker pipeline.

```text
src/ml_comp/
├── augmentations.py    # Copy-Paste, Scene Mixing, and related transforms
├── data.py             # Dataset and manifest utilities
├── inference.py        # Prediction and detection-merging logic
├── metrics.py          # Recall, FAR, matching, and evaluation helpers
└── sampler.py          # Fair Training and class-aware sampling
```

Keep command-line parsing and notebook-specific display code outside this package. Public functions should have clear inputs, outputs, type hints, and tests once the corresponding component is implemented.
