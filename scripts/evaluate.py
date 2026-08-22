"""Evaluate every curated YOLO checkpoint and export per-class metrics.

Datasets and checkpoints are read-only. One JSON cache is written per checkpoint
so an interrupted multi-checkpoint run can resume without repeating completed
GPU work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
PATH_PATTERN = re.compile(
    r"^(?P<label_set>original_labels|new_labels)/"
    r"(?P<model>[^/]+)/(?P<fold>fold\d+)/weights/(?P<checkpoint>[^/]+\.pt)$"
)

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


@dataclass(frozen=True)
class CheckpointSpec:
    path: Path
    relative_path: Path
    label_set: str
    model: str
    fold: str
    checkpoint: str

    @property
    def fold_number(self) -> int:
        return int(self.fold.removeprefix("fold"))

    @property
    def key(self) -> str:
        return f"{self.label_set}__{self.model}__{self.fold}__{self.path.stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate curated checkpoints on matching fold validation sets."
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=ROOT / "models" / "trained",
    )
    parser.add_argument(
        "--original-data-dir",
        type=Path,
        required=True,
        help="Directory containing original-label data_foldN YAML files.",
    )
    parser.add_argument(
        "--new-data-dir",
        type=Path,
        required=True,
        help="Directory containing new-label data_foldN YAML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "trained" / "metrics",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "runs" / "checkpoint_per_class_metrics",
        help="Ignored directory for resumable caches and framework run files.",
    )
    parser.add_argument(
        "--data-yaml-template",
        default="data_fold{fold}_autodl.yaml",
    )
    parser.add_argument("--image-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default=None, help="For example: 0 or cpu")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--framework-revision",
        default="unknown",
        help="Git revision or source-tree hash for the YOLO implementation.",
    )
    parser.add_argument("--expected-checkpoints", type=int, default=48)
    parser.add_argument("--expected-classes", type=int, default=25)
    parser.add_argument(
        "--max-map-delta",
        type=float,
        default=0.005,
        help="Maximum absolute rerun-vs-selection delta for mAP metrics.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore compatible caches and recompute all checkpoints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N checkpoints for a smoke test.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inventory_checkpoints(weights_root: Path) -> list[CheckpointSpec]:
    if not weights_root.is_dir():
        raise FileNotFoundError(f"Weights root does not exist: {weights_root}")

    specs: list[CheckpointSpec] = []
    for path in sorted(weights_root.glob("*/*/fold*/weights/*.pt")):
        relative_path = path.relative_to(weights_root)
        match = PATH_PATTERN.fullmatch(relative_path.as_posix())
        if match is None:
            raise ValueError(f"Unexpected checkpoint path: {relative_path}")
        if path.stat().st_size < 1024:
            raise ValueError(f"Unresolved Git LFS pointer: {path}")
        specs.append(
            CheckpointSpec(
                path=path.resolve(),
                relative_path=relative_path,
                **match.groupdict(),
            )
        )
    if not specs:
        raise ValueError(f"No curated checkpoints found under {weights_root}")
    checkpoint_order = {
        "map_best.pt": 0,
        "precision_fdr_best.pt": 1,
        "recall_best.pt": 2,
    }
    specs.sort(
        key=lambda spec: (
            spec.label_set,
            spec.fold_number,
            spec.model,
            checkpoint_order.get(spec.checkpoint, 99),
        )
    )
    return specs


def resolve_data_yaml(
    spec: CheckpointSpec,
    data_dirs: dict[str, Path],
    template: str,
) -> Path:
    path = data_dirs[spec.label_set] / template.format(fold=spec.fold_number)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found for {spec.relative_path}: {path}")
    return path.resolve()


def resolve_validation_manifest(data_yaml: Path) -> tuple[Path, str]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    validation_source = config.get("val")
    if not isinstance(validation_source, str):
        raise ValueError(f"Expected one validation text manifest in {data_yaml}")

    dataset_root = Path(config.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    manifest = Path(validation_source)
    if not manifest.is_absolute():
        manifest = dataset_root / manifest
    if manifest.suffix.lower() != ".txt" or not manifest.is_file():
        raise ValueError(f"Validation source must be an existing .txt file: {manifest}")
    return manifest.resolve(), validation_source


def manifest_images(manifest: Path) -> list[Path]:
    images: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = (manifest.parent / path).resolve()
        images.append(path)
    if not images:
        raise ValueError(f"Validation manifest is empty: {manifest}")
    return images


def image_to_label_path(image: Path) -> Path:
    parts = list(image.parts)
    positions = [index for index, part in enumerate(parts) if part == "images"]
    if not positions:
        raise ValueError(f"Image path has no images component: {image}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def labels_profile(images: list[Path]) -> tuple[str, int]:
    digest = hashlib.sha256()
    validation_instances = 0
    for image in images:
        label = image_to_label_path(image)
        if not label.is_file():
            raise FileNotFoundError(f"Validation label does not exist: {label}")
        rows = [
            line.strip()
            for line in label.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        numeric_rows = {
            tuple(float(value) for value in row.split())
            for row in rows
        }
        validation_instances += len(numeric_rows)
        label_parts = label.parts
        label_index = max(
            index for index, part in enumerate(label_parts) if part == "labels"
        )
        logical_name = Path(*label_parts[label_index:]).as_posix()
        digest.update(logical_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(label.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), validation_instances


def dataset_identity(data_yaml: Path) -> dict[str, Any]:
    manifest, configured_source = resolve_validation_manifest(data_yaml)
    images = manifest_images(manifest)
    labels_sha256, validation_instances = labels_profile(images)
    return {
        "config_name": data_yaml.name,
        "config_sha256": sha256_file(data_yaml),
        "validation_source": configured_source,
        "validation_manifest_sha256": sha256_file(manifest),
        "validation_labels_sha256": labels_sha256,
        "validation_images": len(images),
        "validation_instances": validation_instances,
    }


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def infer_selected_epoch(checkpoint: dict[str, Any]) -> int | None:
    target = checkpoint.get("train_metrics") or {}
    history = checkpoint.get("train_results") or {}
    epochs = history.get("epoch") if isinstance(history, dict) else None
    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    if not epochs or not all(key in history and key in target for key in keys):
        epoch = checkpoint.get("epoch")
        return int(epoch) + 1 if isinstance(epoch, int) and epoch >= 0 else None

    best_index = min(
        range(len(epochs)),
        key=lambda index: sum(
            abs(float(history[key][index]) - float(target[key])) for key in keys
        ),
    )
    max_delta = max(
        abs(float(history[key][best_index]) - float(target[key])) for key in keys
    )
    return int(epochs[best_index]) if max_delta <= 0.001 else None


def checkpoint_metadata(model: YOLO) -> dict[str, Any]:
    checkpoint = getattr(model, "ckpt", None) or {}
    train_args = checkpoint.get("train_args") or {}
    selection = checkpoint.get("train_metrics") or {}
    return {
        "selected_epoch": infer_selected_epoch(checkpoint),
        "training_date": checkpoint.get("date"),
        "training_run": train_args.get("name"),
        "training_data_config": Path(str(train_args.get("data", ""))).name or None,
        "checkpoint_ultralytics_version": checkpoint.get("version"),
        "selection": {
            "precision": metric_value(selection, "metrics/precision(B)"),
            "recall": metric_value(selection, "metrics/recall(B)"),
            "map50": metric_value(selection, "metrics/mAP50(B)"),
            "map50_95": metric_value(selection, "metrics/mAP50-95(B)"),
        },
    }


def extract_per_class(metrics: Any) -> list[dict[str, Any]]:
    names = {int(key): str(value) for key, value in metrics.names.items()}
    positions = {
        int(class_id): index
        for index, class_id in enumerate(metrics.ap_class_index)
    }
    rows: list[dict[str, Any]] = []
    for class_id, class_name in sorted(names.items()):
        index = positions.get(class_id)
        if index is None:
            precision = recall = f1 = map50 = map50_95 = None
        else:
            precision, recall, map50, map50_95 = (
                float(value) for value in metrics.class_result(index)
            )
            f1 = float(metrics.box.f1[index])
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "images": int(metrics.nt_per_image[class_id]),
                "instances": int(metrics.nt_per_class[class_id]),
                "precision": precision,
                "recall": recall,
                "fdr": None if precision is None else 1.0 - precision,
                "f1": f1,
                "map50": map50,
                "map50_95": map50_95,
            }
        )
    return rows


def evaluate_checkpoint(
    spec: CheckpointSpec,
    data_yaml: Path,
    data_id: dict[str, Any],
    checkpoint_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = YOLO(str(spec.path))
    metadata = checkpoint_metadata(model)
    training_data = metadata.get("training_data_config")
    if training_data and training_data != data_yaml.name:
        raise ValueError(
            f"{spec.relative_path} was trained with {training_data}, not {data_yaml.name}"
        )

    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        iou=args.iou,
        max_det=args.max_det,
        half=False,
        augment=False,
        plots=False,
        save=False,
        save_json=False,
        verbose=False,
        project=str(args.work_dir / "ultralytics"),
        name=spec.key,
        exist_ok=True,
    )
    runtime_seconds = time.perf_counter() - started
    values = metrics.results_dict
    precision = float(values["metrics/precision(B)"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {
            "label_set": spec.label_set,
            "model": spec.model,
            "fold": spec.fold,
            "name": spec.checkpoint,
            "path": spec.relative_path.as_posix(),
            "sha256": checkpoint_sha256,
        },
        "dataset": data_id,
        "metadata": metadata,
        "overall": {
            "precision": precision,
            "recall": float(values["metrics/recall(B)"]),
            "fdr": 1.0 - precision,
            "f1": float(metrics.box.f1.mean()),
            "map50": float(values["metrics/mAP50(B)"]),
            "map50_95": float(values["metrics/mAP50-95(B)"]),
        },
        "per_class": extract_per_class(metrics),
        "runtime_seconds": runtime_seconds,
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }
    del metrics
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def cache_path(work_dir: Path, spec: CheckpointSpec) -> Path:
    return (
        work_dir
        / "cache"
        / spec.label_set
        / spec.model
        / spec.fold
        / f"{spec.path.stem}.json"
    )


def cache_identity(
    spec: CheckpointSpec,
    checkpoint_sha256: str,
    data_id: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    return sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_path": spec.relative_path.as_posix(),
            "checkpoint_sha256": checkpoint_sha256,
            "dataset": data_id,
            "protocol": {
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "iou": args.iou,
                "max_det": args.max_det,
                "half": False,
                "augment": False,
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "framework_revision": args.framework_revision,
            },
        }
    )


def load_or_evaluate(
    spec: CheckpointSpec,
    data_yaml: Path,
    data_id: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bool]:
    checkpoint_sha256 = sha256_file(spec.path)
    identity = cache_identity(spec, checkpoint_sha256, data_id, args)
    path = cache_path(args.work_dir, spec)
    if path.is_file() and not args.force:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache_identity") == identity:
            return cached["result"], True

    result = evaluate_checkpoint(
        spec,
        data_yaml,
        data_id,
        checkpoint_sha256,
        args,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"cache_identity": identity, "result": result},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return result, False


def validate_result(
    result: dict[str, Any],
    expected_classes: int,
    max_map_delta: float,
) -> None:
    rows = result["per_class"]
    if expected_classes > 0 and len(rows) != expected_classes:
        raise ValueError(
            f"{result['checkpoint']['path']} produced {len(rows)} classes; "
            f"expected {expected_classes}"
        )
    class_ids = [row["class_id"] for row in rows]
    if len(class_ids) != len(set(class_ids)):
        raise ValueError(f"Duplicate class IDs in {result['checkpoint']['path']}")
    actual_instances = sum(row["instances"] for row in rows)
    expected_instances = result["dataset"]["validation_instances"]
    if actual_instances <= 0:
        raise ValueError(f"No validation targets for {result['checkpoint']['path']}")
    if actual_instances != expected_instances:
        raise ValueError(
            f"{result['checkpoint']['path']} evaluated {actual_instances} targets; "
            f"dataset fingerprint contains {expected_instances}"
        )

    selection = result["metadata"]["selection"]
    deltas = [
        abs(result["overall"][key] - selection[key])
        for key in ("map50", "map50_95")
        if selection.get(key) is not None
    ]
    if deltas and max(deltas) > max_map_delta:
        raise ValueError(
            f"{result['checkpoint']['path']} rerun mAP delta {max(deltas):.6f} "
            f"exceeds {max_map_delta:.6f}"
        )


def checkpoint_row(result: dict[str, Any]) -> dict[str, Any]:
    checkpoint = result["checkpoint"]
    metadata = result["metadata"]
    selection = metadata["selection"]
    overall = result["overall"]
    speed = result["speed_ms_per_image"]

    def delta(key: str) -> float | None:
        return (
            None
            if selection.get(key) is None
            else overall[key] - selection[key]
        )

    return {
        "label_set": checkpoint["label_set"],
        "model": checkpoint["model"],
        "fold": checkpoint["fold"],
        "checkpoint": checkpoint["name"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "selected_epoch": metadata["selected_epoch"],
        "training_date": metadata["training_date"],
        "training_run": metadata["training_run"],
        "training_data_config": metadata["training_data_config"],
        "checkpoint_ultralytics_version": metadata[
            "checkpoint_ultralytics_version"
        ],
        "evaluation_data_config": result["dataset"]["config_name"],
        "validation_images": result["dataset"]["validation_images"],
        "validation_instances": sum(
            item["instances"] for item in result["per_class"]
        ),
        **overall,
        "selection_precision": selection["precision"],
        "selection_recall": selection["recall"],
        "selection_fdr": (
            None
            if selection["precision"] is None
            else 1.0 - selection["precision"]
        ),
        "selection_map50": selection["map50"],
        "selection_map50_95": selection["map50_95"],
        "delta_precision": delta("precision"),
        "delta_recall": delta("recall"),
        "delta_map50": delta("map50"),
        "delta_map50_95": delta("map50_95"),
        "runtime_seconds": result["runtime_seconds"],
        "preprocess_ms_per_image": speed.get("preprocess"),
        "inference_ms_per_image": speed.get("inference"),
        "postprocess_ms_per_image": speed.get("postprocess"),
    }


def per_class_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoint = result["checkpoint"]
    prefix = {
        "label_set": checkpoint["label_set"],
        "model": checkpoint["model"],
        "fold": checkpoint["fold"],
        "checkpoint": checkpoint["name"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
    }
    return [{**prefix, **row} for row in result["per_class"]]


def decimal(value: Any, places: int = 8) -> str:
    if value is None:
        return ""
    return f"{value:.{places}f}" if isinstance(value, float) else str(value)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: decimal(row.get(field)) for field in fieldnames}
            )


def environment(args: argparse.Namespace) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        primary_device = str(args.device).split(",", maxsplit=1)[0]
        device: int | torch.device
        if primary_device.isdigit():
            device = int(primary_device)
        else:
            device = torch.device(primary_device)
        gpu = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "framework_revision": args.framework_revision,
        "device": args.device,
        "gpu": gpu,
    }


def build_manifest(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    started_at: datetime,
) -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    for result in results:
        key = (
            f"{result['checkpoint']['label_set']}/"
            f"{result['checkpoint']['fold']}"
        )
        datasets[key] = result["dataset"]
    class_names = {
        str(row["class_id"]): row["class_name"]
        for row in results[0]["per_class"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "checkpoint_count": len(results),
        "class_count": len(class_names),
        "per_class_row_count": sum(
            len(result["per_class"]) for result in results
        ),
        "class_names": class_names,
        "protocol": {
            "task": "detect",
            "split": "val",
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "iou": args.iou,
            "max_det": args.max_det,
            "half": False,
            "augment": False,
            "plots": False,
            "fdr_definition": "1 - precision",
            "precision_recall_operating_point": (
                "Ultralytics global confidence threshold maximizing "
                "smoothed mean F1"
            ),
            "map50_95_iou_range": "0.50:0.05:0.95",
        },
        "environment": environment(args),
        "datasets": datasets,
        "artifacts": {
            "checkpoint_metrics": "checkpoints.csv",
            "consolidated_per_class_metrics": "per_class.csv",
            "per_checkpoint_metrics": "by_checkpoint/",
        },
    }


def md_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.5f}"


def write_readme(
    path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    protocol = manifest["protocol"]
    runtime = manifest["environment"]
    max_abs_deltas = {
        key: max(
            (
                abs(float(row[key]))
                for row in rows
                if row.get(key) is not None
            ),
            default=0.0,
        )
        for key in (
            "delta_precision",
            "delta_recall",
            "delta_map50",
            "delta_map50_95",
        )
    }

    lines = [
        "# Per-class checkpoint metrics",
        "",
        (
            f"Validation metrics for all {manifest['checkpoint_count']} curated "
            f"checkpoints and all {manifest['class_count']} target classes "
            f"({manifest['per_class_row_count']} checkpoint-class rows)."
        ),
        "",
        "## Files",
        "",
        "- [per_class.csv](per_class.csv): consolidated checkpoint-class rows.",
        "- [checkpoints.csv](checkpoints.csv): aggregate rerun metrics and signed deltas from embedded selection metrics.",
        "- [by_checkpoint/](by_checkpoint/): one 25-row CSV per checkpoint.",
        "- [manifest.json](manifest.json): protocol, environment, dataset fingerprints, and artifact counts.",
        "",
        "## Protocol",
        "",
        "- Every checkpoint uses the validation manifest for its own label version and fold.",
        (
            f"- Standard Ultralytics detection validation: image size "
            f"{protocol['image_size']}, NMS IoU {protocol['iou']}, max detections "
            f"{protocol['max_det']}, FP32, no test-time augmentation."
        ),
        "- Precision, Recall, and F1 use the single global confidence threshold maximizing the smoothed mean F1 curve.",
        "- FDR equals 1 minus Precision.",
        "- mAP50 is AP at IoU 0.50; mAP50-95 averages AP over IoU 0.50:0.05:0.95.",
        "- Images and Instances count validation images containing the class and its ground-truth objects, respectively.",
        "- Values are per-fold validation results, not test metrics or cross-validation averages.",
        "",
        (
            "Aggregate values are fresh reruns. Maximum absolute deltas from "
            "embedded selection metrics: "
            f"Precision {max_abs_deltas['delta_precision']:.5f}, "
            f"Recall {max_abs_deltas['delta_recall']:.5f}, "
            f"mAP50 {max_abs_deltas['delta_map50']:.5f}, and "
            f"mAP50-95 {max_abs_deltas['delta_map50_95']:.5f}."
        ),
        "",
        "Precision and Recall are recomputed at the global max-F1 confidence operating point and can move more than AP; checkpoints.csv preserves both sets of values and their signed deltas.",
        "",
        "## Reproduction",
        "",
        "    python scripts/evaluate.py --original-data-dir /path/to/original-label-data --new-data-dir /path/to/new-label-data --work-dir runs/checkpoint_per_class_metrics --image-size 800 --batch-size 64 --workers 16 --device 0 --framework-revision SOURCE_ID",
        "",
        "Environment used for the committed report:",
        "",
        f"- Python {runtime['python']}",
        f"- PyTorch {runtime['torch']} with CUDA {runtime['cuda']}",
        f"- Ultralytics {runtime['ultralytics']}",
        f"- GPU {runtime['gpu']}",
        f"- YOLO source identifier {runtime['framework_revision']}",
        "",
        "## Checkpoint index",
        "",
        "| Labels | Model | Fold | Checkpoint | Per-class CSV | Precision | Recall | FDR | mAP50 | mAP50-95 |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        stem = Path(row["checkpoint"]).stem
        checkpoint_link = (
            f"../{row['label_set']}/{row['model']}/{row['fold']}/"
            f"weights/{row['checkpoint']}"
        )
        metrics_link = (
            f"by_checkpoint/{row['label_set']}/{row['model']}/"
            f"{row['fold']}/{stem}.csv"
        )
        cells = [
            row["label_set"],
            row["model"],
            row["fold"].removeprefix("fold"),
            f"[{row['checkpoint']}]({checkpoint_link})",
            f"[CSV]({metrics_link})",
            md_metric(row["precision"]),
            md_metric(row["recall"]),
            md_metric(row["fdr"]),
            md_metric(row["map50"]),
            md_metric(row["map50_95"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    started_at: datetime,
) -> None:
    checkpoint_rows = [checkpoint_row(result) for result in results]
    all_class_rows = [
        row for result in results for row in per_class_rows(result)
    ]
    expected_rows = len(results) * args.expected_classes
    if args.expected_classes > 0 and len(all_class_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} per-class rows, got {len(all_class_rows)}"
        )

    write_csv(
        args.output_dir / "checkpoints.csv",
        CHECKPOINT_FIELDS,
        checkpoint_rows,
    )
    write_csv(
        args.output_dir / "per_class.csv",
        PER_CLASS_FIELDS,
        all_class_rows,
    )
    for result in results:
        checkpoint = result["checkpoint"]
        target = (
            args.output_dir
            / "by_checkpoint"
            / checkpoint["label_set"]
            / checkpoint["model"]
            / checkpoint["fold"]
            / f"{Path(checkpoint['name']).stem}.csv"
        )
        write_csv(target, PER_CLASS_FIELDS, per_class_rows(result))

    manifest = build_manifest(results, args, started_at)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(
        args.output_dir / "README.md",
        checkpoint_rows,
        manifest,
    )


def main() -> None:
    args = parse_args()
    args.weights_root = args.weights_root.resolve()
    args.original_data_dir = args.original_data_dir.resolve()
    args.new_data_dir = args.new_data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    started_at = datetime.now(timezone.utc)
    specs = inventory_checkpoints(args.weights_root)
    if (
        args.expected_checkpoints > 0
        and len(specs) != args.expected_checkpoints
    ):
        raise ValueError(
            f"Found {len(specs)} checkpoints; "
            f"expected {args.expected_checkpoints}"
        )
    if args.limit > 0:
        specs = specs[: args.limit]

    data_dirs = {
        "original_labels": args.original_data_dir,
        "new_labels": args.new_data_dir,
    }
    data_cache: dict[Path, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        data_yaml = resolve_data_yaml(
            spec,
            data_dirs,
            args.data_yaml_template,
        )
        if data_yaml not in data_cache:
            data_cache[data_yaml] = dataset_identity(data_yaml)
        print(f"[{index}/{len(specs)}] {spec.relative_path}", flush=True)
        result, reused = load_or_evaluate(
            spec,
            data_yaml,
            data_cache[data_yaml],
            args,
        )
        validate_result(
            result,
            args.expected_classes,
            args.max_map_delta,
        )
        state = "cache" if reused else f"{result['runtime_seconds']:.1f}s"
        print(
            f"    {state}: mAP50-95={result['overall']['map50_95']:.5f}",
            flush=True,
        )
        results.append(result)

    materialize(results, args, started_at)
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    row_count = sum(len(result["per_class"]) for result in results)
    print(
        f"Wrote {len(results)} checkpoint reports and {row_count} "
        f"per-class rows to {args.output_dir} in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
