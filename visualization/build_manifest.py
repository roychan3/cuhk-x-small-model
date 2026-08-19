#!/usr/bin/env python3
"""Build a compact CUHK-X clip manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from visualization.dataset import (  # noqa: E402
    build_dataset_manifest,
    resolve_dataset_root,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Directory containing Training/ and Testing/. Defaults to "
            "$CUHKX_DATASET_ROOT, then to ~/AAI/opt-scratch/small-model."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "cuhkx_manifest.parquet",
        help="Output .parquet, .csv, or .json path.",
    )
    parser.add_argument(
        "--no-deep-test",
        action="store_true",
        help="Skip reading test CSV/JSON payloads for quality flags.",
    )
    parser.add_argument(
        "--deep-train",
        action="store_true",
        help="Read training CSV/JSON payloads when the source is merged or extracted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    records, sources = build_dataset_manifest(
        dataset_root,
        deep_test=not args.no_deep_test,
        deep_train=args.deep_train,
    )
    if not sources:
        print(f"No dataset sources found under {dataset_root}", file=sys.stderr)
        return 2
    output = write_manifest(records, args.output)
    source_summary = ", ".join(f"{split}={source.kind}" for split, source in sources.items())
    print(f"Wrote {len(records):,} clips to {output}")
    print(f"Sources: {source_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
