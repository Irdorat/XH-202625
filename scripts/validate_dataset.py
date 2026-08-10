#!/usr/bin/env python3
"""Validate source images and YOLO annotations without modifying them."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NUM_CLASSES = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("runs/dataset_audit/v1"))
    parser.add_argument("--duplicate-iou", type=float, default=0.95)
    parser.add_argument("--boundary-tolerance", type=float, default=2e-6)
    parser.add_argument("--edge-threshold", type=float, default=0.001)
    return parser.parse_args()


def add_issue(
    issues: list[dict[str, object]], stem: str, issue: str, detail: str,
    image_path: Path | None = None, label_path: Path | None = None,
    line_number: int | None = None,
) -> None:
    issues.append({
        "stem": stem,
        "issue_type": issue,
        "line_number": line_number if line_number is not None else "",
        "detail": detail,
        "image_path": str(image_path) if image_path else "",
        "label_path": str(label_path) if label_path else "",
    })


def xyxy(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, width, height = values
    return x - width / 2, y - height / 2, x + width / 2, y + height / 2


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = xyxy(a)  # type: ignore[arg-type]
    bx1, by1, bx2, by2 = xyxy(b)  # type: ignore[arg-type]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    image_dir = args.data_root / "images" / args.split
    label_dir = args.data_root / "labels" / args.split
    args.output.mkdir(parents=True, exist_ok=True)

    images = {
        path.stem: path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {
        path.stem: path for path in label_dir.glob("*.txt")
        if not path.name.endswith(":Zone.Identifier")
    }
    issues: list[dict[str, object]] = []
    edge_candidates: list[dict[str, object]] = []

    for stem in sorted(images.keys() - labels.keys()):
        add_issue(issues, stem, "MISSING_LABEL", "Image has no matching annotation", images[stem])
    for stem in sorted(labels.keys() - images.keys()):
        add_issue(issues, stem, "MISSING_IMAGE", "Annotation has no matching image", label_path=labels[stem])

    for stem, image_path in sorted(images.items()):
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image.load()
        except Exception as exc:
            add_issue(issues, stem, "CORRUPTED_IMAGE", f"{type(exc).__name__}: {exc}", image_path)

    for stem, label_path in sorted(labels.items()):
        image_path = images.get(stem)
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            add_issue(issues, stem, "UNREADABLE_LABEL", f"{type(exc).__name__}: {exc}", image_path, label_path)
            continue
        nonempty_lines = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]
        if not nonempty_lines:
            add_issue(issues, stem, "EMPTY_LABEL", "Annotation contains no objects", image_path, label_path)
            continue

        boxes: list[tuple[int, int, tuple[float, float, float, float]]] = []
        for line_number, line in nonempty_lines:
            fields = line.split()
            if len(fields) != 5:
                add_issue(issues, stem, "INVALID_FIELD_COUNT", f"Expected 5 fields, found {len(fields)}: {line}", image_path, label_path, line_number)
                continue
            try:
                class_id = int(fields[0])
                values = tuple(float(value) for value in fields[1:])
            except ValueError:
                add_issue(issues, stem, "NON_NUMERIC_VALUE", f"Cannot parse: {line}", image_path, label_path, line_number)
                continue
            if not 0 <= class_id < NUM_CLASSES:
                add_issue(issues, stem, "INVALID_CLASS", f"class_id={class_id}; expected 0..{NUM_CLASSES - 1}", image_path, label_path, line_number)
            if not all(math.isfinite(value) for value in values):
                add_issue(issues, stem, "NON_FINITE_COORDINATE", f"coordinates={values}", image_path, label_path, line_number)
                continue
            x, y, width, height = values
            if width <= 0 or height <= 0:
                add_issue(issues, stem, "NON_POSITIVE_SIZE", f"width={width}, height={height}", image_path, label_path, line_number)
                continue
            if not all(0 <= value <= 1 for value in values):
                add_issue(issues, stem, "COORDINATE_OUT_OF_RANGE", f"coordinates={values}", image_path, label_path, line_number)
            x1, y1, x2, y2 = xyxy(values)  # type: ignore[arg-type]
            edge_distances = {
                "left": x1,
                "top": y1,
                "right": 1 - x2,
                "bottom": 1 - y2,
            }
            touching = [
                side for side, distance in edge_distances.items()
                if distance <= args.edge_threshold
            ]
            if touching:
                edge_candidates.append({
                    "stem": stem,
                    "line_number": line_number,
                    "class_id": class_id,
                    "touching_edges": "|".join(touching),
                    "min_edge_distance": min(edge_distances.values()),
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                    "image_path": str(image_path) if image_path else "",
                    "label_path": str(label_path),
                })
            if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                excursion = max(-x1, -y1, x2 - 1, y2 - 1)
                issue = (
                    "BOX_BOUNDARY_ROUNDING"
                    if excursion <= args.boundary_tolerance
                    else "BOX_OUT_OF_BOUNDS"
                )
                add_issue(issues, stem, issue, f"xyxy=({x1:.9f}, {y1:.9f}, {x2:.9f}, {y2:.9f}); excursion={excursion:.9g}", image_path, label_path, line_number)
            boxes.append((line_number, class_id, values))  # type: ignore[arg-type]

        for index, (line_a, class_a, box_a) in enumerate(boxes):
            for line_b, class_b, box_b in boxes[index + 1:]:
                if class_a != class_b:
                    continue
                overlap = iou(box_a, box_b)
                if box_a == box_b:
                    issue = "EXACT_DUPLICATE_BOX"
                elif overlap >= args.duplicate_iou:
                    issue = "NEAR_DUPLICATE_BOX"
                else:
                    continue
                add_issue(issues, stem, issue, f"lines={line_a},{line_b}; class_id={class_a}; IoU={overlap:.6f}", image_path, label_path, line_b)

    fields = ["stem", "issue_type", "line_number", "detail", "image_path", "label_path"]
    with (args.output / "automatic_issues.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)

    edge_fields = [
        "stem", "line_number", "class_id", "touching_edges", "min_edge_distance",
        "x_min", "y_min", "x_max", "y_max", "image_path", "label_path",
    ]
    with (args.output / "edge_box_candidates.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=edge_fields)
        writer.writeheader()
        writer.writerows(edge_candidates)

    counts = Counter(str(issue["issue_type"]) for issue in issues)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "images_checked": len(images),
        "labels_checked": len(labels),
        "files_with_issues": len({str(issue["stem"]) for issue in issues}),
        "total_issues": len(issues),
        "duplicate_iou_threshold": args.duplicate_iou,
        "boundary_tolerance": args.boundary_tolerance,
        "edge_threshold": args.edge_threshold,
        "edge_box_candidates": len(edge_candidates),
        "images_with_edge_boxes": len({str(row["stem"]) for row in edge_candidates}),
        "issues_by_type": dict(sorted(counts.items())),
    }
    (args.output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
