#!/usr/bin/env python3
"""Remap MAR20 aircraft annotations to the project's class IDs 4 through 23."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_SOURCE_NAMES = [
    "B-1B", "B-52", "C-130", "C-17", "C-5", "E-3", "E-8", "F-15", "F-16", "F-22",
    "FA-18", "KC-10", "KC-135", "P-3C", "SU-24", "SU-34", "SU-35", "TU-160", "TU-22", "TU-95",
]

PROJECT_NAMES = [
    "HM", "LQS", "QHS", "MS", "A1_SU-35", "A2_C-130", "A3_C-17", "A4_C-5",
    "A5_F-16", "A6_TU-160", "A7_E-3", "A8_B-52", "A9_P-3C", "A10_B-1B",
    "A11_E-8", "A12_TU-22", "A13_F-15", "A14_KC-135", "A15_F-22", "A16_FA-18",
    "A17_TU-95", "A18_KC-10", "A19_SU-34", "A20_SU-24", "FSC",
]

SOURCE_TO_TARGET = {
    "SU-35": 4,
    "C-130": 5,
    "C-17": 6,
    "C-5": 7,
    "F-16": 8,
    "TU-160": 9,
    "E-3": 10,
    "B-52": 11,
    "P-3C": 12,
    "B-1B": 13,
    "E-8": 14,
    "TU-22": 15,
    "F-15": 16,
    "KC-135": 17,
    "F-22": 18,
    "FA-18": 19,
    "TU-95": 20,
    "KC-10": 21,
    "SU-34": 22,
    "SU-24": 23,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/extra_dataset/MAR20"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/mar20_20class"))
    parser.add_argument(
        "--exclude-reference",
        type=Path,
        default=Path("data/raw/images"),
        help="Exclude MAR20 source IDs already present below this reference directory.",
    )
    parser.add_argument(
        "--exclude-id-file",
        type=Path,
        default=Path("configs/baseline_v2/mar20_excluded_source_ids.txt"),
        help="Additional source IDs confirmed as duplicate scenes under different filenames.",
    )
    parser.add_argument(
        "--keep-all-augmentations",
        action="store_true",
        help="Keep all Roboflow variants instead of one deterministic variant per source image.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_source_names(config_path: Path) -> list[str]:
    values: dict[str, object] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line[:1].isspace():
            continue
        key, value = line.split(":", 1)
        if key in {"nc", "names"}:
            values[key] = ast.literal_eval(value.strip())
    names = values.get("names")
    if names != EXPECTED_SOURCE_NAMES or values.get("nc") != 20:
        raise ValueError(f"Unexpected MAR20 taxonomy in {config_path}: {names!r}")
    return names


def source_stem(path: Path) -> str:
    """Return the pre-augmentation Roboflow image identifier."""
    return path.stem.split(".rf.", 1)[0].removesuffix("_jpg").removesuffix("_bmp")


def reference_mar20_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for path in root.rglob("MAR20_*"):
        if not path.is_file() or path.name.endswith(":Zone.Identifier"):
            continue
        ids.add(path.stem.removeprefix("MAR20_"))
    return ids


def load_excluded_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_names = load_source_names(source / "data.yaml")

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
        shutil.rmtree(output)
    images_output = output / "images"
    labels_output = output / "labels"
    images_output.mkdir(parents=True)
    labels_output.mkdir(parents=True)

    images = sorted(
        path for path in (source / "images").iterdir()
        if path.is_file() and not path.name.endswith(":Zone.Identifier")
    )
    groups: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        groups[source_stem(image_path)].append(image_path)
    reference_ids = reference_mar20_ids(args.exclude_reference.resolve())
    audited_ids = load_excluded_ids(args.exclude_id_file.resolve())
    excluded_ids = reference_ids | audited_ids
    eligible_images = [path for path in images if source_stem(path) not in excluded_ids]
    eligible_groups: dict[str, list[Path]] = defaultdict(list)
    for image_path in eligible_images:
        eligible_groups[source_stem(image_path)].append(image_path)
    selected = (
        eligible_images
        if args.keep_all_augmentations
        else [paths[0] for _, paths in sorted(eligible_groups.items())]
    )

    counts: Counter[str] = Counter()
    for image_path in selected:
        label_path = source / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}")
        remapped_rows: list[str] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
            source_id = int(fields[0])
            if not 0 <= source_id < len(source_names):
                raise ValueError(f"{label_path}:{line_number}: invalid class ID {source_id}")
            source_name = source_names[source_id]
            target_id = SOURCE_TO_TARGET[source_name]
            remapped_rows.append(" ".join((str(target_id), *fields[1:])))
            counts[source_name] += 1

        if not remapped_rows:
            raise ValueError(f"Unexpected empty annotation: {label_path}")
        os.symlink(os.path.relpath(image_path, images_output), images_output / image_path.name)
        (labels_output / label_path.name).write_text("\n".join(remapped_rows) + "\n", encoding="utf-8")

    names_yaml = "\n".join(f"  {class_id}: {name}" for class_id, name in enumerate(PROJECT_NAMES))
    (output / "data.yaml").write_text(
        f"path: {output}\ntrain: images\nnc: 25\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )
    report = {
        "source": str(source),
        "output": str(output),
        "source_export_images": len(images),
        "unique_source_images": len(groups),
        "reference_root": str(args.exclude_reference.resolve()),
        "reference_mar20_ids": len(reference_ids),
        "audited_duplicate_ids": sorted(audited_ids),
        "excluded_overlapping_source_images": len(groups.keys() & reference_ids),
        "excluded_audited_duplicate_images": len(groups.keys() & audited_ids),
        "eligible_unique_source_images": len(eligible_groups),
        "kept_images": len(selected),
        "discarded_augmented_variants": len(eligible_images) - len(selected),
        "keep_all_augmentations": args.keep_all_augmentations,
        "objects_by_source_class": dict(sorted(counts.items())),
        "source_to_target_id": SOURCE_TO_TARGET,
    }
    (output / "remap_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
