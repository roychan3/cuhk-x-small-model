#!/usr/bin/env python3
"""Compare standardized validation reports from registered algorithms.

The row and column format lives in ``visualization/comparison_format.py`` so
that this CLI and the dashboard's ``Algorithm comparison`` page cannot drift.
That module is standard-library only, so importing it here does not pull
Streamlit, pandas, or Plotly into the CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from visualization.comparison_format import COMPARISON_FIELDS, assign_labels, comparison_row


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _row(path: Path) -> dict[str, object]:
    return comparison_row(path, json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    args = parse_args()
    reports = args.reports or sorted((REPOSITORY_ROOT / "artifacts").glob("*/validation.json"))
    # Labels are assigned in discovery order, before sorting, so the same set of
    # reports always gets the same labels whichever metric they end up sorted by.
    rows = assign_labels([_row(path) for path in reports if path.is_file()])
    if not rows:
        raise SystemExit("No validation reports found")
    rows.sort(key=lambda row: float(row["macro_f1"] or -1), reverse=True)
    fields = COMPARISON_FIELDS
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
