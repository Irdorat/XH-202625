# Data

This directory separates immutable source data from reproducible split definitions.

```text
data/
├── raw/       # Organizer-provided images and labels
└── splits/    # Versioned train/validation manifests
```

Rules:

- never modify, rename, or move files inside `raw/`;
- do not commit `raw/` or generated dataset archives to Git;
- keep small split manifests in Git;
- place generated or augmented data outside `raw/` and document how it was produced.

See [splits/README.md](splits/README.md) for split conventions.
