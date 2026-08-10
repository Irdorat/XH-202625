#!/usr/bin/env python3
"""Create a reproducible inventory of the immutable source dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CLASS_NAMES = [
    "HM", "LQS", "QHS", "MS", "A1_SU-35", "A2_C-130", "A3_C-17",
    "A4_C-5", "A5_F-16", "A6_TU-160", "A7_E-3", "A8_B-52",
    "A9_P-3C", "A10_B-1B", "A11_E-8", "A12_TU-22", "A13_F-15",
    "A14_KC-135", "A15_F-22", "A16_FA-18", "A17_TU-95", "A18_KC-10",
    "A19_SU-34", "A20_SU-24", "FSC",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    """Read PNG/JPEG dimensions without third-party dependencies."""
    with path.open("rb") as source:
        header = source.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", header[16:24])
        if header[:2] != b"\xff\xd8":
            raise ValueError(f"Unsupported image format: {path}")
        source.seek(2)
        while True:
            byte = source.read(1)
            if not byte:
                raise ValueError(f"JPEG dimensions not found: {path}")
            if byte != b"\xff":
                continue
            marker = source.read(1)
            while marker == b"\xff":
                marker = source.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_raw = source.read(2)
            if len(length_raw) != 2:
                raise ValueError(f"Truncated JPEG: {path}")
            length = struct.unpack(">H", length_raw)[0]
            if marker and marker[0] in {
                0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
            }:
                dimensions = source.read(5)
                height, width = struct.unpack(">HH", dimensions[1:5])
                return width, height
            source.seek(length - 2, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("runs/dataset_audit/v1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.data_root / "images" / args.split
    label_dir = args.data_root / "labels" / args.split
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    images = {
        path.stem: path for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {
        path.stem: path for path in sorted(label_dir.glob("*.txt"))
        if not path.name.endswith(":Zone.Identifier")
    }
    stems = sorted(images.keys() | labels.keys())
    class_counts: Counter[int] = Counter()
    annotated_images: Counter[int] = Counter()
    inventory: list[dict[str, object]] = []
    invalid_rows = 0

    for stem in stems:
        image_path = images.get(stem)
        label_path = labels.get(stem)
        width = height = None
        if image_path:
            width, height = image_size(image_path)

        object_count = 0
        classes_in_image: set[int] = set()
        if label_path:
            for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
                fields = line.split()
                try:
                    if len(fields) != 5:
                        raise ValueError
                    class_id = int(fields[0])
                    [float(value) for value in fields[1:]]
                    if not 0 <= class_id < len(CLASS_NAMES):
                        raise ValueError
                except ValueError:
                    invalid_rows += 1
                    continue
                object_count += 1
                class_counts[class_id] += 1
                classes_in_image.add(class_id)
        for class_id in classes_in_image:
            annotated_images[class_id] += 1

        inventory.append({
            "stem": stem,
            "image_path": str(image_path) if image_path else "",
            "label_path": str(label_path) if label_path else "",
            "width": width if width is not None else "",
            "height": height if height is not None else "",
            "image_bytes": image_path.stat().st_size if image_path else "",
            "label_bytes": label_path.stat().st_size if label_path else "",
            "object_count": object_count,
            "image_sha256": sha256(image_path) if image_path else "",
            "label_sha256": sha256(label_path) if label_path else "",
        })

    with (output / "file_inventory.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    with (output / "class_statistics.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=["class_id", "class_name", "object_count", "image_count"]
        )
        writer.writeheader()
        for class_id, class_name in enumerate(CLASS_NAMES):
            writer.writerow({
                "class_id": class_id,
                "class_name": class_name,
                "object_count": class_counts[class_id],
                "image_count": annotated_images[class_id],
            })

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "split": args.split,
        "image_count": len(images),
        "label_count": len(labels),
        "paired_count": len(images.keys() & labels.keys()),
        "images_without_labels": len(images.keys() - labels.keys()),
        "labels_without_images": len(labels.keys() - images.keys()),
        "object_count": sum(class_counts.values()),
        "invalid_annotation_rows": invalid_rows,
        "class_count": len(CLASS_NAMES),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
