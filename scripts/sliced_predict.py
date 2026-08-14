"""Run tiled object detection on one large image and merge the predictions.

Example launch from the repository root::

python scripts/sliced_predict.py \
  --source data/test/ocean_scene_10000x10000.png \
  --weights runs/YOLO11n/yolo11n_fold1_img800_base_b4w2/weights/best_0.92475.pt \
  --tile-size 800 \
  --overlap 0.20 \
  --device 0 
"""

import argparse
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import batched_nms
from ultralytics import RTDETR, YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "sliced_predict"
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

# Legitimate competition images may exceed Pillow's default decompression warning limit.
Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slice one large image, run an Ultralytics detector on every tile, "
            "merge overlapping detections, and render boxes on the original image."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="Input image path.")
    parser.add_argument("--weights", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument(
        "--model-type",
        choices=("auto", "yolo", "rtdetr"),
        default="auto",
        help="Detector loader. Auto recognizes RT-DETR from the checkpoint path.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=800)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Model input size. Defaults to tile-size.",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--tile-iou", type=float, default=0.70)
    parser.add_argument("--global-iou", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--device", default=None, help="For example: 0, 1, or cpu.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.source}")
    if args.source.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported input image format: {args.source.suffix}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {args.weights}")
    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap must be in the range [0, 1)")
    for name in ("confidence", "tile_iou", "global_iou"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name.replace('_', '-')} must be in the range [0, 1]")


def synchronize_device(device: str) -> None:
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()


def calculate_positions(image_size: int, tile_size: int, stride: int) -> list[int]:
    if image_size <= tile_size:
        return [0]
    positions = list(range(0, image_size - tile_size + 1, stride))
    last_position = image_size - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def build_tiles(
    image_width: int,
    image_height: int,
    tile_size: int,
    stride: int,
) -> list[dict[str, int]]:
    x_positions = calculate_positions(image_width, tile_size, stride)
    y_positions = calculate_positions(image_height, tile_size, stride)
    return [
        {
            "tile_id": tile_id,
            "x1": x,
            "y1": y,
            "x2": min(x + tile_size, image_width),
            "y2": min(y + tile_size, image_height),
        }
        for tile_id, (y, x) in enumerate(
            (y, x) for y in y_positions for x in x_positions
        )
    ]


def iter_image_tiles(
    image_path: Path,
    coordinates: list[dict[str, int]],
) -> Iterator[tuple[Image.Image, dict[str, int]]]:
    with Image.open(image_path) as image:
        for tile_info in coordinates:
            box = tuple(tile_info[key] for key in ("x1", "y1", "x2", "y2"))
            yield image.crop(box), tile_info


def resolve_model_type(requested: str, weights: Path) -> str:
    if requested != "auto":
        return requested
    normalized_path = str(weights).lower().replace("_", "-")
    return "rtdetr" if "rtdetr" in normalized_path or "rt-detr" in normalized_path else "yolo"


def load_model(weights: Path, model_type: str) -> tuple[Any, str]:
    resolved_type = resolve_model_type(model_type, weights)
    model_class = RTDETR if resolved_type == "rtdetr" else YOLO
    return model_class(str(weights)), resolved_type


def predict_tile(
    model: Any,
    tile: Image.Image,
    tile_info: dict[str, int],
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    result = model.predict(
        source=tile,
        imgsz=args.image_size or args.tile_size,
        conf=args.confidence,
        iou=args.tile_iou,
        max_det=args.max_detections,
        device=args.device,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.int().detach().cpu().numpy()
    predictions: list[dict[str, Any]] = []

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        local_x1, local_y1, local_x2, local_y2 = map(float, box)
        if local_x2 <= local_x1 or local_y2 <= local_y1:
            continue
        global_bbox = [
            max(0.0, min(local_x1 + tile_info["x1"], image_width)),
            max(0.0, min(local_y1 + tile_info["y1"], image_height)),
            max(0.0, min(local_x2 + tile_info["x1"], image_width)),
            max(0.0, min(local_y2 + tile_info["y1"], image_height)),
        ]
        predictions.append(
            {
                "tile_id": tile_info["tile_id"],
                "tile_x": tile_info["x1"],
                "tile_y": tile_info["y1"],
                "local_bbox": [local_x1, local_y1, local_x2, local_y2],
                "global_bbox": global_bbox,
                "class_id": int(class_id),
                "class_name": result.names[int(class_id)],
                "confidence": float(confidence),
            }
        )
    return predictions


def classwise_nms(
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    if not predictions:
        return []
    boxes = torch.tensor([item["global_bbox"] for item in predictions], dtype=torch.float32)
    scores = torch.tensor([item["confidence"] for item in predictions], dtype=torch.float32)
    classes = torch.tensor([item["class_id"] for item in predictions], dtype=torch.int64)
    keep = batched_nms(boxes, scores, classes, iou_threshold).cpu().tolist()
    return [predictions[index] for index in keep]


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = (
        (255, 64, 64),
        (64, 220, 64),
        (64, 160, 255),
        (255, 200, 64),
        (220, 64, 255),
        (64, 255, 220),
    )
    return palette[class_id % len(palette)]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def render_predictions(
    source: Path,
    predictions: list[dict[str, Any]],
) -> Image.Image:
    with Image.open(source) as image:
        rendered = image.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(max(rendered.size) / 850))
    font = load_font(max(14, round(max(rendered.size) / 310)))

    for prediction in predictions:
        x1, y1, x2, y2 = prediction["global_bbox"]
        color = class_color(prediction["class_id"])
        label = f'{prediction["class_name"]} {prediction["confidence"]:.2f}'
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 8)
        draw.rectangle(
            (x1, label_y, x1 + text_width + 10, label_y + text_height + 6), fill=color
        )
        draw.text((x1 + 5, label_y + 2), label, fill=(0, 0, 0), font=font)
    return rendered


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.source = args.source.resolve()
    args.weights = args.weights.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    stride = max(1, round(args.tile_size * (1 - args.overlap)))

    output_image = args.output / f"{args.source.stem}_predictions.png"
    predictions_path = args.output / f"{args.source.stem}_predictions.json"
    timing_path = args.output / f"{args.source.stem}_timing.json"

    pipeline_started_at = time.perf_counter()
    model, model_type = load_model(args.weights, args.model_type)

    with Image.open(args.source) as source:
        image_width, image_height = source.size
    tiles = build_tiles(image_width, image_height, args.tile_size, stride)

    raw_predictions: list[dict[str, Any]] = []
    tile_seconds: list[float] = []
    for index, (tile, tile_info) in enumerate(iter_image_tiles(args.source, tiles), start=1):
        synchronize_device(args.device)
        tile_started_at = time.perf_counter()
        raw_predictions.extend(
            predict_tile(model, tile, tile_info, image_width, image_height, args)
        )
        synchronize_device(args.device)
        tile_seconds.append(time.perf_counter() - tile_started_at)
        print(
            f"\rTiles: {index}/{len(tiles)} | raw detections: {len(raw_predictions)}",
            end="",
            flush=True,
        )
    print()

    predictions_ready_at = time.perf_counter()
    final_predictions = classwise_nms(raw_predictions, args.global_iou)
    nms_finished_at = time.perf_counter()

    rendered = render_predictions(args.source, final_predictions)
    rendered.save(output_image, format="PNG", compress_level=4)
    write_json(predictions_path, final_predictions)
    pipeline_finished_at = time.perf_counter()

    timing_array = np.asarray(tile_seconds, dtype=np.float64)
    summary = {
        "source": str(args.source),
        "weights": str(args.weights),
        "model_type": model_type,
        "device": str(args.device),
        "image_size": [image_width, image_height],
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "stride": stride,
        "tile_count": len(tiles),
        "raw_detection_count": len(raw_predictions),
        "final_detection_count": len(final_predictions),
        "confidence_threshold": args.confidence,
        "global_nms_iou": args.global_iou,
        "seconds_to_raw_predictions": predictions_ready_at - pipeline_started_at,
        "global_nms_seconds": nms_finished_at - predictions_ready_at,
        "render_and_save_seconds": pipeline_finished_at - nms_finished_at,
        "total_pipeline_seconds": pipeline_finished_at - pipeline_started_at,
        "mean_tile_seconds": float(timing_array.mean()),
        "median_tile_seconds": float(np.median(timing_array)),
        "p95_tile_seconds": float(np.percentile(timing_array, 95)),
    }
    write_json(timing_path, summary)

    print(f"Final detections: {len(final_predictions)}")
    print(f"Total pipeline time: {summary['total_pipeline_seconds']:.3f} s")
    print(f"Rendered image: {output_image}")
    print(f"Predictions: {predictions_path}")
    print(f"Timing: {timing_path}")


if __name__ == "__main__":
    main()
