# Docker

This directory will contain the reproducible competition inference container.

```text
docker/
├── Dockerfile
├── entrypoint.sh
└── README.md
```

The final image must install runtime dependencies, load the selected checkpoint, accept the organizer-defined input, and produce results in the required format.

Keep training, notebooks, datasets, and unnecessary experiment artifacts out of the image. Document the build command, run command, input/output contract, CUDA requirements, and end-to-end timing procedure when the organizer publishes the final interface.
