#!/usr/bin/env python3
"""Download FAIR1M dataset from HuggingFace to data/extra_dataset/FAIR1M.

Uses snapshot_download to avoid the datasets 5.x XML parser bug.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/extra_dataset/FAIR1M"),
        help="Where to save the dataset",
    )
    parser.add_argument(
        "--repo-id",
        default="blanchon/FAIR1M",
        help="HuggingFace dataset repo ID",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} → {output}")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output),
    )
    print(f"Done: {path}")
    print("\nContents:")
    for p in sorted(Path(path).iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
