#!/usr/bin/env python3
"""Curate final YOLO26 five-fold artifacts into a non-overwriting release tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "yolo26n": {"batch": 64},
    "yolo26s": {"batch": 32},
}
CLASS_NAMES = [
    "HM",
    "LQS",
    "QHS",
    "MS",
    "A1_SU-35",
    "A2_C-130",
    "A3_C-17",
    "A4_C-5",
    "A5_F-16",
    "A6_TU-160",
    "A7_E-3",
    "A8_B-52",
    "A9_P-3C",
    "A10_B-1B",
    "A11_E-8",
    "A12_TU-22",
    "A13_F-15",
    "A14_KC-135",
    "A15_F-22",
    "A16_FA-18",
    "A17_TU-95",
    "A18_KC-10",
    "A19_SU-34",
    "A20_SU-24",
    "FSC",
]
RUN_ARTIFACTS = (
    "args.yaml",
    "results.csv",
    "training_complete.json",
    "results.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
VALIDATING = re.compile(
    r"Validating .*?/(yolo26[ns])_final_fold([0-4])_.*?/weights/best\.pt"
)
METRIC_ROW = re.compile(
    r"^\s*(\S+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s*$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_path(source_root: Path, model: str, fold: int) -> Path:
    batch = MODELS[model]["batch"]
    return (
        source_root
        / f"runs_final_dataset_fold{fold}_u84126"
        / f"{model}_final_fold{fold}_img800_musgd_lr005_b{batch}w16_u84126"
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_per_class_metrics(logs: list[Path]) -> list[dict]:
    collected: dict[tuple[str, int, str], dict] = {}
    class_ids = {name: index for index, name in enumerate(CLASS_NAMES)}
    current: tuple[str, int] | None = None

    for log in logs:
        if not log.is_file():
            raise FileNotFoundError(log)
        clean = ANSI_ESCAPE.sub("", log.read_text(encoding="utf-8", errors="replace"))
        for line in clean.splitlines():
            validating = VALIDATING.search(line)
            if validating:
                current = (validating.group(1), int(validating.group(2)))
                continue
            metric = METRIC_ROW.match(line)
            if current is None or metric is None:
                continue
            class_name = metric.group(1)
            if class_name not in class_ids:
                continue
            precision = float(metric.group(4))
            recall = float(metric.group(5))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            row = {
                "model": current[0],
                "fold": current[1],
                "class_id": class_ids[class_name],
                "class_name": class_name,
                "images": int(metric.group(2)),
                "instances": int(metric.group(3)),
                "precision": precision,
                "recall": recall,
                "fdr": round(1.0 - precision, 6),
                "f1": round(f1, 6),
                "map50": float(metric.group(6)),
                "map50_95": float(metric.group(7)),
            }
            collected[(current[0], current[1], class_name)] = row

    expected = len(MODELS) * 5 * len(CLASS_NAMES)
    if len(collected) != expected:
        missing = [
            (model, fold, class_name)
            for model in MODELS
            for fold in range(5)
            for class_name in CLASS_NAMES
            if (model, fold, class_name) not in collected
        ]
        raise RuntimeError(
            f"Expected {expected} per-class rows, found {len(collected)}; "
            f"missing first entries: {missing[:10]}"
        )
    return sorted(
        collected.values(), key=lambda row: (row["model"], row["fold"], row["class_id"])
    )


def build_readme(aggregate: list[dict], focus: list[dict], total_images: int) -> str:
    grouped = {
        model: [row for row in aggregate if row["model"] == model] for model in MODELS
    }
    lines = [
        "# Final dataset YOLO26 results (2026-08-28)",
        "",
        "This directory is an additive result set. It does not replace the existing",
        "`original_labels/` or `new_labels/` collections.",
        "",
        "## Protocol",
        "",
        f"- Dataset pool: {total_images} images, 25 classes",
        "- Existing fold0-fold4 validation membership; every other final-dataset image is training data",
        "- Image size: 800",
        "- Optimizer: MuSGD (`lr0=0.005`)",
        "- Maximum epochs: 150; early-stopping patience: 50",
        "- YOLO26n batch: 64; YOLO26s batch: 32",
        "- Ultralytics: 8.4.126 from `/root/miniconda3/lib/python3.10/site-packages/ultralytics`",
        "- Selection: each run's Ultralytics `best.pt`",
        "",
        "## Five-fold aggregate",
        "",
        "| Model | Precision | Recall | mAP50 | mAP50-95 | mAP50-95 std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, rows in grouped.items():
        lines.append(
            "| {model} | {precision:.5f} | {recall:.5f} | {map50:.5f} | "
            "{map50_95:.5f} | {std:.5f} |".format(
                model=model,
                precision=statistics.mean(row["precision"] for row in rows),
                recall=statistics.mean(row["recall"] for row in rows),
                map50=statistics.mean(row["map50"] for row in rows),
                map50_95=statistics.mean(row["map50_95"] for row in rows),
                std=statistics.pstdev(row["map50_95"] for row in rows),
            )
        )

    by_model_fold = {(row["model"], row["fold"]): row for row in aggregate}
    lines.extend(
        [
            "",
            "## Fold results",
            "",
            "| Fold | YOLO26n mAP50-95 | YOLO26s mAP50-95 | S minus N |",
            "|---:|---:|---:|---:|",
        ]
    )
    for fold in range(5):
        n_value = by_model_fold[("yolo26n", fold)]["map50_95"]
        s_value = by_model_fold[("yolo26s", fold)]["map50_95"]
        lines.append(f"| {fold} | {n_value:.5f} | {s_value:.5f} | {s_value - n_value:+.5f} |")

    focus_lookup = {
        (row["model"], row["class_name"]): row for row in focus if row["fold"] == 4
    }
    lines.extend(
        [
            "",
            "## Fold4 MS and FSC",
            "",
            "| Class | Model | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for class_name in ("MS", "FSC"):
        for model in MODELS:
            row = focus_lookup[(model, class_name)]
            lines.append(
                f"| {class_name} | {model} | {row['images']} | {row['instances']} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['map50']:.3f} | {row['map50_95']:.3f} |"
            )

    lines.extend(
        [
            "",
            "YOLO26s outperformed YOLO26n on mAP50-95 in every matched fold. FSC",
            "remains the principal long-tail weakness: fold4 contains 93 FSC instances",
            "in only 13 validation images, and FSC objects are substantially smaller than MS.",
            "",
            "## Contents",
            "",
            "- `aggregate_metrics.csv`: one row for each of the 10 completed runs",
            "- `per_class_metrics.csv`: 250 rows (10 runs x 25 classes)",
            "- `fold4_ms_fsc_metrics.csv`: focused MS/FSC comparison",
            "- `dataset_splits.csv`: train/validation image and box counts",
            "- `manifest.json` and `SHA256SUMS`: provenance and file integrity",
            "- `<model>/fold<k>/weights/best.pt`: selected checkpoint tracked by Git LFS",
            "- Each fold also contains `args.yaml`, `results.csv`, result curves, and confusion matrices",
            "",
            "## Comparability note",
            "",
            "Validation membership follows the prior fold basename lists, but the final dataset",
            "contains revised annotations and a small number of revised image files. Metrics here",
            "must not be treated as a strict same-annotation comparison with older result sets.",
            "",
        ]
    )
    return "\n".join(lines)


def package(source_root: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {destination}")

    dataset = source_root / "datasets/final_dataset_20260828"
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)

    aggregate: list[dict] = []
    split_rows: list[dict] = []
    source_runs: list[dict] = []

    # Validate every required source before creating the destination.
    for model in MODELS:
        for fold in range(5):
            run = run_path(source_root, model, fold)
            required = [run / name for name in RUN_ARTIFACTS] + [run / "weights/best.pt"]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing source artifacts: {missing}")
            complete = json.loads((run / "training_complete.json").read_text())
            if complete.get("status") != "complete":
                raise RuntimeError(f"Run is not complete: {run}")

    per_class = parse_per_class_metrics(
        [
            source_root / "final_fold0_yolo26n_u84126.log",
            source_root / "final_fold0_yolo26s_u84126.log",
            source_root / "yolo26ns_remaining_folds_u84126_queue.log",
        ]
    )

    destination.mkdir(parents=True)
    for model in MODELS:
        for fold in range(5):
            run = run_path(source_root, model, fold)
            output = destination / model / f"fold{fold}"
            output.mkdir(parents=True)
            for name in RUN_ARTIFACTS:
                shutil.copy2(run / name, output / name)
            (output / "weights").mkdir()
            shutil.copy2(run / "weights/best.pt", output / "weights/best.pt")

            complete = json.loads((run / "training_complete.json").read_text())
            weight = output / "weights/best.pt"
            precision = float(complete["precision"])
            recall = float(complete["recall"])
            aggregate.append(
                {
                    "model": model,
                    "fold": fold,
                    "train_images": int(complete["train_images"]),
                    "validation_images": int(complete["validation_images"]),
                    "epochs_completed": int(complete["epochs_completed"]),
                    "best_epoch": int(complete["best_map50_95_epoch"]),
                    "precision": precision,
                    "recall": recall,
                    "fdr": round(1.0 - precision, 6),
                    "f1": round(2 * precision * recall / (precision + recall), 6),
                    "map50": float(complete["map50"]),
                    "map50_95": float(complete["map50_95"]),
                    "weight_size_bytes": weight.stat().st_size,
                    "weight_sha256": sha256_file(weight),
                }
            )
            source_runs.append(
                {
                    "model": model,
                    "fold": fold,
                    "source": str(run),
                    "packaged": str(output.relative_to(destination)),
                }
            )

    for fold in range(5):
        report_path = dataset / f"splits/fold{fold}/split_report.json"
        report = json.loads(report_path.read_text())
        split_rows.append(
            {
                "fold": fold,
                "total_images": int(report["total_images"]),
                "train_images": int(report["train"]["images"]),
                "train_boxes": sum(report["train"]["class_box_counts"].values()),
                "validation_images": int(report["validation"]["images"]),
                "validation_boxes": sum(report["validation"]["class_box_counts"].values()),
                "source_validation_list_sha256": report["source_validation_list_sha256"],
                "packaged_validation_list_sha256": report["validation_list_sha256"],
            }
        )

    aggregate_fields = list(aggregate[0])
    write_csv(destination / "aggregate_metrics.csv", aggregate_fields, aggregate)
    per_class_fields = list(per_class[0])
    write_csv(destination / "per_class_metrics.csv", per_class_fields, per_class)
    focus = [row for row in per_class if row["fold"] == 4 and row["class_name"] in {"MS", "FSC"}]
    write_csv(destination / "fold4_ms_fsc_metrics.csv", per_class_fields, focus)
    write_csv(destination / "dataset_splits.csv", list(split_rows[0]), split_rows)

    total_images = split_rows[0]["total_images"]
    (destination / "README.md").write_text(
        build_readme(aggregate, focus, total_images), encoding="utf-8"
    )

    files_before_manifest = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        "schema": "xh202625-final-yolo-results-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset),
        "dataset_total_images": total_images,
        "classes": CLASS_NAMES,
        "environment": {
            "python": "/root/miniconda3/bin/python",
            "ultralytics_version": "8.4.126",
            "ultralytics_path": "/root/miniconda3/lib/python3.10/site-packages/ultralytics/__init__.py",
            "training_gpu": "NVIDIA GeForce RTX 5090",
        },
        "source_runs": source_runs,
        "files": [
            {
                "path": str(path.relative_to(destination)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files_before_manifest
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    checksum_files = sorted(path for path in destination.rglob("*") if path.is_file())
    (destination / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(destination)}\n"
            for path in checksum_files
            if path.name != "SHA256SUMS"
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "destination": str(destination),
                "runs": len(aggregate),
                "per_class_rows": len(per_class),
                "files": sum(1 for path in destination.rglob("*") if path.is_file()),
                "bytes": sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    package(args.source_root, args.destination)
