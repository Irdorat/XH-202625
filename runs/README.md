# Runs

This directory contains generated outputs from training, validation, and inference.

Typical run contents include:

- resolved experiment configuration;
- aggregate and per-class metrics;
- plots and validation artifacts;
- logs and timing measurements;
- intermediate and best checkpoints.

Run outputs are not committed to Git. Preserve important runs in versioned archives, and copy only the selected checkpoint to `models/trained/`. Use descriptive names such as `baseline_v2_control`, `baseline_v2_fair_training`, or `baseline_v2_copy_paste`.
