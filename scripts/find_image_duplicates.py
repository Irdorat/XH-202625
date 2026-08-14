#!/usr/bin/env python3
"""Find exact and perceptually similar images across two dataset directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/raw/images"))
    parser.add_argument("--candidate", type=Path, default=Path("data/generated/mar20_20class/images"))
    parser.add_argument("--output", type=Path, default=Path("runs/dataset_audit/mar20_duplicates"))
    parser.add_argument(
        "--reference-prefix",
        default="",
        help="Prefix to remove from reference stems before matching source IDs, e.g. MAR20_.",
    )
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=6,
        help="Maximum 64-bit dHash Hamming distance to report; lower is stricter.",
    )
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.endswith(":Zone.Identifier")
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        get_pixels = getattr(resized, "get_flattened_data", resized.getdata)
        pixels = list(get_pixels())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def normalized_source_id(path: Path, reference_prefix: str = "") -> str:
    stem = path.stem
    if reference_prefix and stem.startswith(reference_prefix):
        stem = stem[len(reference_prefix):]
    return stem.split("_jpg.rf.", 1)[0].split("_bmp.rf.", 1)[0]


def main() -> None:
    args = parse_args()
    if not 0 <= args.dhash_threshold <= 64:
        raise ValueError("--dhash-threshold must be between 0 and 64")
    reference_root = args.reference.resolve()
    candidate_root = args.candidate.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference_paths = image_paths(reference_root)
    candidate_paths = image_paths(candidate_root)
    if not reference_paths or not candidate_paths:
        raise ValueError(
            f"Both inputs must contain images: reference={len(reference_paths)}, candidate={len(candidate_paths)}"
        )

    unreadable: list[dict[str, str]] = []
    reference_hashes: list[tuple[Path, int]] = []
    candidate_hashes: list[tuple[Path, int]] = []
    for collection, destination in (
        (reference_paths, reference_hashes),
        (candidate_paths, candidate_hashes),
    ):
        for path in collection:
            try:
                destination.append((path, dhash(path)))
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                unreadable.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    reference_sha: dict[str, list[Path]] = defaultdict(list)
    for path in reference_paths:
        reference_sha[sha256(path)].append(path)
    exact_rows: list[dict[str, object]] = []
    for path in candidate_paths:
        digest = sha256(path)
        for reference_path in reference_sha.get(digest, []):
            exact_rows.append(
                {
                    "reference_path": str(reference_path),
                    "candidate_path": str(path),
                    "sha256": digest,
                }
            )

    reference_ids: dict[str, list[Path]] = defaultdict(list)
    for path in reference_paths:
        reference_ids[normalized_source_id(path, args.reference_prefix)].append(path)
    filename_rows: list[dict[str, str]] = []
    for path in candidate_paths:
        source_id = normalized_source_id(path)
        for reference_path in reference_ids.get(source_id, []):
            filename_rows.append(
                {
                    "source_id": source_id,
                    "reference_path": str(reference_path),
                    "candidate_path": str(path),
                }
            )

    perceptual_rows: list[dict[str, object]] = []
    for candidate_path, candidate_hash in candidate_hashes:
        best_distance = 65
        best_references: list[Path] = []
        for reference_path, reference_hash in reference_hashes:
            distance = (candidate_hash ^ reference_hash).bit_count()
            if distance < best_distance:
                best_distance = distance
                best_references = [reference_path]
            elif distance == best_distance:
                best_references.append(reference_path)
        if best_distance <= args.dhash_threshold:
            for reference_path in best_references:
                perceptual_rows.append(
                    {
                        "reference_path": str(reference_path),
                        "candidate_path": str(candidate_path),
                        "dhash_distance": best_distance,
                    }
                )

    with (output / "exact_duplicates.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["reference_path", "candidate_path", "sha256"])
        writer.writeheader()
        writer.writerows(exact_rows)
    with (output / "perceptual_candidates.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=["reference_path", "candidate_path", "dhash_distance"]
        )
        writer.writeheader()
        writer.writerows(sorted(perceptual_rows, key=lambda row: int(row["dhash_distance"])))
    with (output / "source_id_matches.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=["source_id", "reference_path", "candidate_path"]
        )
        writer.writeheader()
        writer.writerows(filename_rows)

    summary = {
        "reference_root": str(reference_root),
        "candidate_root": str(candidate_root),
        "reference_images": len(reference_paths),
        "candidate_images": len(candidate_paths),
        "exact_duplicate_pairs": len(exact_rows),
        "source_id_match_pairs": len(filename_rows),
        "reference_prefix": args.reference_prefix,
        "perceptual_threshold": args.dhash_threshold,
        "perceptual_candidate_pairs": len(perceptual_rows),
        "unreadable_images": unreadable,
        "note": "Perceptual candidates require visual review; dHash is not proof that two files show the same source scene.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
