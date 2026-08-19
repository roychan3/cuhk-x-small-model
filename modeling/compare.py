#!/usr/bin/env python3
"""Compare standardized validation reports from registered algorithms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _selected_metrics(report: dict[str, Any]) -> dict[str, float]:
    selected = report["selected_parameters"]
    for candidate in report.get("aggregate_metrics", []):
        if candidate.get("parameters") == selected:
            return candidate["metrics"]
    return {}


def _row(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = _selected_metrics(report)
    return {
        "algorithm": report.get("algorithm", path.parent.name),
        "parameters": json.dumps(report.get("selected_parameters", {}), sort_keys=True),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "training_clips": report.get("training_clips"),
        "report": str(path),
    }


def main() -> int:
    args = parse_args()
    reports = args.reports or sorted((REPOSITORY_ROOT / "artifacts").glob("*/validation.json"))
    rows = [_row(path) for path in reports if path.is_file()]
    if not rows:
        raise SystemExit("No validation reports found")
    rows.sort(key=lambda row: float(row["macro_f1"] or -1), reverse=True)
    fields = (
        "algorithm",
        "parameters",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "training_clips",
        "report",
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved comparison to {args.output}")
    else:
        print("\t".join(fields))
        for row in rows:
            print("\t".join(str(row[field]) for field in fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
