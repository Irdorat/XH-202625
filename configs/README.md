# Configurations

This directory stores versioned dataset and experiment configurations.

```text
configs/
├── baseline_v1/    # Historical RT-DETR-L baseline configuration
└── baseline_v2/    # Baseline v2 configuration under development
```

Each baseline directory contains:

- `dataset.yaml` — image manifests, class count, and class names;
- `baseline_v*.yaml` — model, training, augmentation, and output parameters.

Keep paths repository-relative and create a new experiment configuration instead of silently changing the parameters of a completed run.
