"""Run a small, reproducible RT-DETR baseline inference demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs" / "baseline_v1" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "data" / "splits" / "v1" / "demo_val.txt"
DEFAULT_PROJECT = ROOT / "runs" / "baseline_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline v1 on a directory, image, or text manifest."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="demo")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default=None, help="For example: 0 or cpu")
    return parser.parse_args()


def resolve_sources(source: Path, limit: int) -> list[str] | str:
    if not source.exists():
        raise FileNotFoundError(f"Inference source does not exist: {source}")

    if source.suffix.lower() != ".txt":
        return str(source)

    paths: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Image listed in {source} does not exist: {path}")
        paths.append(str(path))

    if not paths:
        raise ValueError(f"No images found in manifest: {source}")
    return paths[:limit] if limit > 0 else paths


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.weights}")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    sources = resolve_sources(args.source, args.limit)
    model = RTDETR(str(args.weights))
    results = model.predict(
        source=sources,
        conf=args.confidence,
        imgsz=args.image_size,
        device=device,
        save=True,
        project=str(args.output),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    source_names = sources if isinstance(sources, list) else None
    summary = []
    for index, result in enumerate(results):
        counts: dict[str, int] = {}
        if result.boxes is not None:
            for class_id in result.boxes.cls.int().cpu().tolist():
                name = result.names[class_id]
                counts[name] = counts.get(name, 0) + 1
        summary.append(
            {
                "image": source_names[index] if source_names else str(result.path),
                "detections": len(result.boxes) if result.boxes is not None else 0,
                "classes": counts,
                "speed_ms": result.speed,
            }
        )

    save_dir = Path(results[0].save_dir)
    summary_path = save_dir / "demo_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved predictions to: {save_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
