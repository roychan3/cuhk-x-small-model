#!/usr/bin/env python3
"""Build a compact CUHK-X clip manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
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
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Optional JSON file updated with background build progress.",
    )
    return parser.parse_args()


def write_progress(path: Path | None, **values: object) -> None:
    if path is None:
        return
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        temporary.write_text(json.dumps(values, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    try:
        def report_progress(phase: str, processed: int, total: int) -> None:
            write_progress(
                args.progress_file,
                state="building",
                phase=phase,
                processed=processed,
                total=total,
            )

        records, sources = build_dataset_manifest(
            dataset_root,
            deep_test=not args.no_deep_test,
            deep_train=args.deep_train,
            progress_callback=report_progress if args.progress_file else None,
        )
        if not sources:
            write_progress(
                args.progress_file,
                state="error",
                message=f"No dataset sources found under {dataset_root}",
            )
            print(f"No dataset sources found under {dataset_root}", file=sys.stderr)
            return 2
        write_progress(
            args.progress_file,
            state="building",
            phase="writing",
            processed=1,
            total=1,
        )
        output = write_manifest(records, args.output)
        write_progress(
            args.progress_file,
            state="complete",
            phase="complete",
            processed=1,
            total=1,
            clips=len(records),
        )
        source_summary = ", ".join(f"{split}={source.kind}" for split, source in sources.items())
        print(f"Wrote {len(records):,} clips to {output}")
        print(f"Sources: {source_summary}")
        return 0
    except Exception as exc:
        write_progress(args.progress_file, state="error", message=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
