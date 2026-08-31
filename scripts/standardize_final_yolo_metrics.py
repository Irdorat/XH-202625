#!/usr/bin/env python3
"""Align final YOLO26 metrics with the branch's per-checkpoint CSV schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


LABEL_SET = "final_dataset_20260828_u84126"
PER_CLASS_FIELDS = [
    "label_set",
    "model",
    "fold",
    "checkpoint",
    "checkpoint_path",
    "checkpoint_sha256",
    "class_id",
    "class_name",
    "images",
    "instances",
    "precision",
    "recall",
    "fdr",
    "f1",
    "map50",
    "map50_95",
]
CHECKPOINT_FIELDS = [
    "label_set",
    "model",
    "fold",
    "checkpoint",
    "checkpoint_path",
    "checkpoint_sha256",
    "selected_epoch",
    "training_date",
    "training_run",
    "training_data_config",
    "checkpoint_ultralytics_version",
    "evaluation_data_config",
    "validation_images",
    "validation_instances",
    "precision",
    "recall",
    "fdr",
    "f1",
    "map50",
    "map50_95",
    "selection_precision",
    "selection_recall",
    "selection_fdr",
    "selection_map50",
    "selection_map50_95",
    "delta_precision",
    "delta_recall",
    "delta_map50",
    "delta_map50_95",
    "runtime_seconds",
    "preprocess_ms_per_image",
    "inference_ms_per_image",
    "postprocess_ms_per_image",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decimal(value: str | float) -> str:
    return f"{float(value):.8f}"


def standardize(root: Path) -> None:
    metrics_dir = root / "metrics"
    if metrics_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing metrics directory: {metrics_dir}")

    raw_per_class = read_csv(root / "per_class_metrics.csv")
    aggregate = read_csv(root / "aggregate_metrics.csv")
    aggregate_lookup = {(row["model"], int(row["fold"])): row for row in aggregate}
    if len(raw_per_class) != 250 or len(aggregate_lookup) != 10:
        raise RuntimeError("Expected 250 per-class rows and 10 aggregate rows")

    standardized: list[dict[str, str]] = []
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for raw in raw_per_class:
        model = raw["model"]
        fold_number = int(raw["fold"])
        aggregate_row = aggregate_lookup[(model, fold_number)]
        checkpoint_path = f"{LABEL_SET}/{model}/fold{fold_number}/weights/best.pt"
        row = {
            "label_set": LABEL_SET,
            "model": model,
            "fold": f"fold{fold_number}",
            "checkpoint": "best.pt",
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": aggregate_row["weight_sha256"],
            "class_id": raw["class_id"],
            "class_name": raw["class_name"],
            "images": raw["images"],
            "instances": raw["instances"],
            "precision": decimal(raw["precision"]),
            "recall": decimal(raw["recall"]),
            "fdr": decimal(raw["fdr"]),
            "f1": decimal(raw["f1"]),
            "map50": decimal(raw["map50"]),
            "map50_95": decimal(raw["map50_95"]),
        }
        standardized.append(row)
        grouped[(model, fold_number)].append(row)

    standardized.sort(key=lambda row: (row["model"], row["fold"], int(row["class_id"])))
    write_csv(root / "per_class_metrics.csv", PER_CLASS_FIELDS, standardized)
    write_csv(metrics_dir / "per_class.csv", PER_CLASS_FIELDS, standardized)

    focus = [
        row
        for row in standardized
        if row["fold"] == "fold4" and row["class_name"] in {"MS", "FSC"}
    ]
    write_csv(root / "fold4_ms_fsc_metrics.csv", PER_CLASS_FIELDS, focus)

    for (model, fold_number), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["class_id"]))
        if len(rows) != 25:
            raise RuntimeError(f"Expected 25 classes for {model} fold{fold_number}")
        write_csv(
            metrics_dir
            / "by_checkpoint"
            / LABEL_SET
            / model
            / f"fold{fold_number}"
            / "best.csv",
            PER_CLASS_FIELDS,
            rows,
        )

    checkpoint_rows: list[dict[str, str]] = []
    for (model, fold_number), aggregate_row in sorted(aggregate_lookup.items()):
        run_dir = root / model / f"fold{fold_number}"
        complete = json.loads((run_dir / "training_complete.json").read_text())
        checkpoint = run_dir / "weights/best.pt"
        precision = float(aggregate_row["precision"])
        recall = float(aggregate_row["recall"])
        f1 = 2 * precision * recall / (precision + recall)
        checkpoint_rows.append(
            {
                "label_set": LABEL_SET,
                "model": model,
                "fold": f"fold{fold_number}",
                "checkpoint": "best.pt",
                "checkpoint_path": f"{LABEL_SET}/{model}/fold{fold_number}/weights/best.pt",
                "checkpoint_sha256": aggregate_row["weight_sha256"],
                "selected_epoch": aggregate_row["best_epoch"],
                "training_date": datetime.fromtimestamp(
                    checkpoint.stat().st_mtime, timezone.utc
                ).isoformat(),
                "training_run": Path(complete["best_checkpoint"]).parents[1].name,
                "training_data_config": Path(complete["data"]).name,
                "checkpoint_ultralytics_version": complete["ultralytics_version"],
                "evaluation_data_config": Path(complete["data"]).name,
                "validation_images": aggregate_row["validation_images"],
                "validation_instances": str(
                    sum(int(row["instances"]) for row in grouped[(model, fold_number)])
                ),
                "precision": decimal(precision),
                "recall": decimal(recall),
                "fdr": decimal(1.0 - precision),
                "f1": decimal(f1),
                "map50": decimal(aggregate_row["map50"]),
                "map50_95": decimal(aggregate_row["map50_95"]),
                "selection_precision": decimal(precision),
                "selection_recall": decimal(recall),
                "selection_fdr": decimal(1.0 - precision),
                "selection_map50": decimal(aggregate_row["map50"]),
                "selection_map50_95": decimal(aggregate_row["map50_95"]),
                "delta_precision": "0.00000000",
                "delta_recall": "0.00000000",
                "delta_map50": "0.00000000",
                "delta_map50_95": "0.00000000",
                "runtime_seconds": "",
                "preprocess_ms_per_image": "",
                "inference_ms_per_image": "",
                "postprocess_ms_per_image": "",
            }
        )
    write_csv(metrics_dir / "checkpoints.csv", CHECKPOINT_FIELDS, checkpoint_rows)

    metrics_readme = """# Per-class checkpoint metrics

Selection-time final validation metrics for 10 checkpoints and all 25 target
classes (250 checkpoint-class rows).

## Files

- `per_class.csv`: consolidated checkpoint-class rows.
- `checkpoints.csv`: aggregate selection-time metrics.
- `by_checkpoint/`: one 25-row CSV per checkpoint.
- `manifest.json`: protocol, provenance, and precision limits.

## Protocol

- Each checkpoint uses its final-dataset validation split for the same fold.
- Ultralytics 8.4.126 detection validation at image size 800.
- Precision, Recall, FDR, F1, mAP50, and mAP50-95 follow the existing branch schema.
- FDR is `1 - Precision` and F1 is the harmonic mean of Precision and Recall.
- These values were parsed from the final `best.pt` validation tables produced
  at training completion; no new validation was run while another authorized
  FSC training job occupied the GPU.
- The source log prints per-class metrics to three decimal places. Values are
  padded to eight decimal places for CSV schema compatibility, not to claim
  additional numerical precision. Aggregate metrics retain their five-decimal
  training-completion values.
"""
    (metrics_dir / "README.md").write_text(metrics_readme, encoding="utf-8")

    metrics_files = sorted(
        path
        for path in metrics_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    metrics_manifest = {
        "schema": "xh202625-per-class-checkpoint-metrics-v1",
        "label_set": LABEL_SET,
        "checkpoints": 10,
        "classes": 25,
        "per_class_rows": 250,
        "source": "selection-time final best.pt validation logs",
        "fresh_rerun": False,
        "per_class_source_precision_decimals": 3,
        "csv_decimal_places": 8,
        "environment": {
            "python": "3.10.12",
            "pytorch": "2.13.0+cu130",
            "cuda": "13.0",
            "ultralytics": "8.4.126",
            "gpu": "NVIDIA GeForce RTX 5090",
        },
        "files": [
            {
                "path": str(path.relative_to(metrics_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in metrics_files
        ],
    }
    (metrics_dir / "manifest.json").write_text(
        json.dumps(metrics_manifest, indent=2) + "\n", encoding="utf-8"
    )

    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    metrics_bullet = "- `metrics/`: branch-compatible consolidated and per-checkpoint class CSVs\n"
    if metrics_bullet not in readme:
        marker = "- `aggregate_metrics.csv`: one row for each of the 10 completed runs\n"
        if marker not in readme:
            raise RuntimeError("README contents marker not found")
        readme = readme.replace(marker, metrics_bullet + marker)
        readme_path.write_text(readme, encoding="utf-8")

    root_manifest_path = root / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text())
    root_manifest["metrics_standardization"] = {
        "schema_matches": "models/trained/metrics/by_checkpoint/*.csv",
        "checkpoints": 10,
        "per_class_rows": 250,
        "source_precision_decimals": 3,
        "csv_decimal_places": 8,
        "fresh_rerun": False,
    }
    inventory = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    )
    root_manifest["files"] = [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in inventory
    ]
    root_manifest_path.write_text(
        json.dumps(root_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    checksum_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in checksum_files
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "root": str(root),
                "checkpoints": len(checkpoint_rows),
                "per_class_rows": len(standardized),
                "by_checkpoint_csvs": len(grouped),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    standardize(args.root)
