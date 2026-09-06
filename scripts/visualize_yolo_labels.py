"""Draw YOLO detection labels on their source images with a distinct color per annotation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ultralytics.utils.plotting import Annotator, colors


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/images/train"), help="Training image directory")
    parser.add_argument("--labels", type=Path, default=Path("data/labels/train"), help="YOLO label directory")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/label_visualization"), help="Directory for annotated images"
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum images to process; default processes all images")
    parser.add_argument("--line-width", type=int, default=None, help="Bounding-box line width; default scales with image size")
    return parser.parse_args()


def read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read one YOLO detection label file and validate its rows."""
    annotations = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 values, got {len(values)}")
        class_id, x_center, y_center, width, height = map(float, values)
        if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
            raise ValueError(f"{label_path}:{line_number}: normalized box values must be in [0, 1]")
        annotations.append((int(class_id), x_center, y_center, width, height))
    return annotations


def draw_labels(image_path: Path, label_path: Path, output_path: Path, line_width: int | None = None) -> None:
    """Draw all labels, assigning every annotation instance a distinct deterministic color."""
    image = np.asarray(Image.open(image_path).convert("RGB"))[:, :, ::-1].copy()
    image_height, image_width = image.shape[:2]
    annotations = read_labels(label_path)
    duplicate_totals = Counter(annotations)
    duplicate_indices = defaultdict(int)
    annotator = Annotator(image, line_width=line_width)

    for index, annotation in enumerate(annotations):
        class_id, x_center, y_center, width, height = annotation
        x1 = max(0, round((x_center - width / 2) * image_width))
        y1 = max(0, round((y_center - height / 2) * image_height))
        x2 = min(image_width - 1, round((x_center + width / 2) * image_width))
        y2 = min(image_height - 1, round((y_center + height / 2) * image_height))
        duplicate_indices[annotation] += 1
        duplicate_suffix = (
            f" dup {duplicate_indices[annotation]}/{duplicate_totals[annotation]}"
            if duplicate_totals[annotation] > 1
            else ""
        )
        annotator.box_label((x1, y1, x2, y2), f"class {class_id}{duplicate_suffix}", color=colors(index, bgr=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(annotator.result()[:, :, ::-1]).save(output_path)


def main() -> None:
    """Visualize labels for matching images and report skipped files."""
    args = parse_args()
    if not args.images.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images}")
    if not args.labels.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {args.labels}")

    image_paths = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    processed = skipped = 0
    for image_path in image_paths:
        relative_path = image_path.relative_to(args.images)
        label_path = args.labels / relative_path.with_suffix(".txt")
        if not label_path.is_file():
            print(f"Skip (missing label): {image_path}")
            skipped += 1
            continue
        output_path = args.output / relative_path.with_suffix(".jpg")
        draw_labels(image_path, label_path, output_path, args.line_width)
        processed += 1

    print(f"Done: {processed} images written to {args.output.resolve()} ({skipped} skipped)")


if __name__ == "__main__":
    main()
