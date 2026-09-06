#!/usr/bin/env python3
"""Remap ShipRSImageNet annotations to the project's four ship classes."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
from collections import Counter
from pathlib import Path

TARGET_NAMES = ["HM", "LQS", "QHS", "MS"]

# Project classes: HM=aircraft carrier, LQS=amphibious ship,
# QHS=destroyer/frigate, MS=merchant ship.
SOURCE_TO_TARGET_NAME = {
    "Enterprise": "HM",
    "Midway": "HM",
    "Nimitz": "HM",
    "Other Aircraft Carrier": "HM",
    "Austin LL": "LQS",
    "LHA LL": "LQS",
    "LSD 41 LL": "LQS",
    "Osumi LL": "LQS",
    "Other Landing": "LQS",
    "Sanantonio AS": "LQS",
    "Wasp LL": "LQS",
    "YuDao LL": "LQS",
    "YuDeng LL": "LQS",
    "YuTing LL": "LQS",
    "YuZhao LL": "LQS",
    "Arleigh Burke DD": "QHS",
    "Asagiri DD": "QHS",
    "Atago DD": "QHS",
    "Hatsuyuki DD": "QHS",
    "Hyuga DD": "QHS",
    "Other Destroyer": "QHS",
    "Other Frigate": "QHS",
    "Other Warship": "QHS",
    "Perry FF": "QHS",
    "Ticonderoga": "QHS",
    "Barge": "MS",
    "Cargo": "MS",
    "Container Ship": "MS",
    "Ferry": "MS",
    "Fishing Vessel": "MS",
    "Motorboat": "MS",
    "Oil Tanker": "MS",
    "Other Merchant": "MS",
    "RoRo": "MS",
    "Sailboat": "MS",
    "Tugboat": "MS",
    "Yacht": "MS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/extra_dataset/ShipRSImage"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/shiprsimage_4class"))
    parser.add_argument(
        "--ambiguous-policy",
        choices=("skip-image", "drop-annotation"),
        default="skip-image",
        help="Skip images containing excluded classes (safe default), or keep images and drop only excluded boxes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_source_names(source: Path) -> list[str]:
    config_path = source / "data.yaml"
    values: dict[str, object] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line[:1].isspace():
            continue
        key, value = line.split(":", 1)
        if key in {"nc", "names"}:
            values[key] = ast.literal_eval(value.strip())
    names = values.get("names")
    if not isinstance(names, list) or len(names) != values.get("nc"):
        raise ValueError(f"Invalid class names in {config_path}")
    return [str(name) for name in names]


def find_image(images_dir: Path, stem: str) -> Path:
    matches = [
        path
        for path in images_dir.glob(f"{stem}.*")
        if path.is_file() and not path.name.endswith(":Zone.Identifier")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one image for {stem}, found {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_names = load_source_names(source)
    target_ids = {name: index for index, name in enumerate(TARGET_NAMES)}
    source_to_target = {
        source_id: target_ids[SOURCE_TO_TARGET_NAME[name]]
        for source_id, name in enumerate(source_names)
        if name in SOURCE_TO_TARGET_NAME
    }

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
        shutil.rmtree(output)
    images_output = output / "images"
    labels_output = output / "labels"
    images_output.mkdir(parents=True)
    labels_output.mkdir(parents=True)

    counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    skipped_images = 0
    kept_images = 0

    label_paths = sorted(
        path
        for path in (source / "labels").glob("*.txt")
        if not path.name.endswith(":Zone.Identifier")
    )
    for label_path in label_paths:
        remapped_rows: list[str] = []
        has_excluded_class = False
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
            source_id = int(fields[0])
            if not 0 <= source_id < len(source_names):
                raise ValueError(f"{label_path}:{line_number}: invalid class ID {source_id}")
            if source_id not in source_to_target:
                has_excluded_class = True
                excluded_counts[source_names[source_id]] += 1
                continue
            target_id = source_to_target[source_id]
            remapped_rows.append(" ".join((str(target_id), *fields[1:])))

        if has_excluded_class and args.ambiguous_policy == "skip-image":
            skipped_images += 1
            continue
        if not remapped_rows:
            skipped_images += 1
            continue

        image_path = find_image(source / "images", label_path.stem)
        os.symlink(os.path.relpath(image_path, images_output), images_output / image_path.name)
        (labels_output / label_path.name).write_text("\n".join(remapped_rows) + "\n", encoding="utf-8")
        kept_images += 1
        for row in remapped_rows:
            counts[TARGET_NAMES[int(row.split(maxsplit=1)[0])]] += 1

    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(TARGET_NAMES))
    (output / "data.yaml").write_text(
        f"path: {output}\ntrain: images\nnc: {len(TARGET_NAMES)}\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )
    report = {
        "source": str(source),
        "output": str(output),
        "ambiguous_policy": args.ambiguous_policy,
        "source_images": len(label_paths),
        "kept_images": kept_images,
        "skipped_images": skipped_images,
        "objects_by_target_class": dict(counts),
        "excluded_objects_by_source_class": dict(sorted(excluded_counts.items())),
        "source_to_target": SOURCE_TO_TARGET_NAME,
    }
    (output / "remap_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
