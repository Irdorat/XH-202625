# Notebooks

This directory is for exploratory analysis, validation, visualization, and error analysis.

```text
notebooks/
└── baseline_v1/
    ├── baseline_v1_analysis.ipynb
    └── baseline_v1_results.ipynb
```

Notebooks may read fixed manifests and checkpoints, but must not mutate `data/raw` or create the authoritative train/validation split. Reusable training, evaluation, augmentation, and inference logic should move to `scripts/` or `src/ml_comp/`.

Before sharing a result, keep the important outputs visible and ensure all explanatory text is in English.
