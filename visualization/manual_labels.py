"""Persistence helpers for resumable manual labeling of test clips."""

from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REQUIRED_COLUMNS = ("path", "prediction")

#: ``csv.DictReader`` collects fields past the header under this key, so a row
#: with too many columns is detectable instead of being silently truncated on
#: the next write.
_EXTRA_FIELDS = "__extra__"


def manual_label_path(test_csv: str | Path) -> Path:
    """Return the sibling CSV used for manual labels.

    For example, ``Testing/test.csv`` becomes
    ``Testing/test_manual_label.csv``.
    """

    source = Path(test_csv).expanduser()
    return source.with_name(f"{source.stem}_manual_label{source.suffix}")


def archive_manual_label_file(output_csv: str | Path) -> Path:
    """Move an unusable manual-label file aside and return its new path.

    Every read path needs this file to parse before the page can render, so a
    corrupt or out-of-order file would otherwise be recoverable only from a
    shell. Renaming instead of deleting keeps whatever labels it still holds.
    """

    target = Path(output_csv).expanduser()
    backup = target.with_name(
        f"{target.stem}.invalid-{uuid.uuid4().hex[:8]}{target.suffix}"
    )
    target.replace(backup)
    return backup


def _validate_paths(rows: Sequence[Mapping[str, object]], source: object) -> None:
    """Reject a path column the loader would refuse to read back."""

    paths = [str(row.get("path", "") or "").strip() for row in rows]
    if any(not value for value in paths):
        raise ValueError(f"Label CSV contains an empty path: {source}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"Label CSV contains duplicate paths: {source}")


def _validate_predictions(
    rows: Sequence[Mapping[str, object]],
    source: object,
    allowed: set[int] | None,
) -> None:
    """Reject prediction values the loader would refuse to read back."""

    for row_number, row in enumerate(rows, start=2):
        value = str(row.get("prediction", "") or "").strip()
        if not value:
            continue
        try:
            action_id = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Manual label on row {row_number} must be an integer "
                f"action_id: {value!r} ({source})"
            ) from exc
        if allowed is not None and action_id not in allowed:
            raise ValueError(
                f"Unknown manual action_id {action_id} on row {row_number}: {source}"
            )


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, restkey=_EXTRA_FIELDS)
        fieldnames = list(reader.fieldnames or ())
        if not set(REQUIRED_COLUMNS) <= set(fieldnames):
            raise ValueError(
                f"Label CSV must contain path and prediction columns: {path}"
            )
        rows: list[dict[str, str]] = []
        for row_number, raw in enumerate(reader, start=2):
            # A ragged row silently loses its overflow on the next write, so
            # refuse it here rather than quietly rewriting a smaller file.
            if _EXTRA_FIELDS in raw:
                raise ValueError(
                    f"Label CSV row {row_number} has more columns than the "
                    f"header: {path}"
                )
            if any(raw.get(field) is None for field in fieldnames):
                raise ValueError(
                    f"Label CSV row {row_number} has fewer columns than the "
                    f"header: {path}"
                )
            rows.append({field: str(raw[field]) for field in fieldnames})

    _validate_paths(rows, path)
    return fieldnames, rows


def load_manual_label_rows(
    test_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    valid_action_ids: Iterable[int] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Load a resumable label table without modifying the source CSV.

    A new table starts with blank predictions even if the source happens to
    contain model predictions. Once the manual-label file exists it becomes the
    source of truth, provided it still describes the same clips in the same
    order as the original test index.
    """

    source_path = Path(test_csv).expanduser()
    target_path = (
        Path(output_csv).expanduser()
        if output_csv is not None
        else manual_label_path(source_path)
    )
    # Aliasing would quietly break the promise above: the source would be read
    # as an existing label table, keeping its model predictions, and the next
    # write would overwrite the test index itself.
    if target_path.resolve() == source_path.resolve():
        raise ValueError(
            f"Manual labels must not be written over the test CSV: {source_path}"
        )
    source_fields, source_rows = _read_rows(source_path)
    resuming = target_path.is_file()
    if resuming:
        output_fields, rows = _read_rows(target_path)
        source_paths = [row["path"].strip() for row in source_rows]
        output_paths = [row["path"].strip() for row in rows]
        if output_paths != source_paths:
            raise ValueError(
                "Existing manual-label CSV does not match the active test CSV "
                f"clip order: {target_path}"
            )
        fieldnames = output_fields
    else:
        fieldnames = source_fields
        rows = [dict(row, prediction="") for row in source_rows]

    allowed = set(valid_action_ids) if valid_action_ids is not None else None
    _validate_predictions(rows, target_path, allowed)
    return fieldnames, rows


def write_manual_label_rows(
    output_csv: str | Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    *,
    valid_action_ids: Iterable[int] | None = None,
) -> None:
    """Atomically write the current manual labels.

    Everything :func:`load_manual_label_rows` checks is checked here first, so
    a table this function accepts is one the loader can reopen. Validating on
    read alone let the dashboard persist a file it then refused to load, with
    no way back except deleting it by hand.
    """

    target = Path(output_csv).expanduser()
    if not set(REQUIRED_COLUMNS) <= set(fieldnames):
        raise ValueError("Manual label output requires path and prediction columns")
    _validate_paths(rows, target)
    _validate_predictions(
        rows,
        target,
        set(valid_action_ids) if valid_action_ids is not None else None,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.stem}.{uuid.uuid4().hex}.tmp{target.suffix}"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in fieldnames} for row in rows
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _is_labeled(row: Mapping[str, object]) -> bool:
    return bool(str(row.get("prediction", "") or "").strip())


def initial_label_clip(
    clip_ids: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> str:
    """Clip the page should open on: the first one still missing a label."""

    if not clip_ids:
        raise ValueError("There are no clips to label")
    for clip_id, row in zip(clip_ids, rows, strict=True):
        if not _is_labeled(row):
            return clip_id
    return clip_ids[0]


def next_unlabeled_clip(
    clip_ids: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    start_index: int,
) -> str:
    """Clip to advance to after labeling ``start_index``.

    The search wraps, so finishing the last clip goes back to gaps left earlier
    in the run. It only considers other clips, so the result always moves even
    when the caller has not yet recorded a label for ``start_index``; once
    nothing is unlabeled it falls back to the immediate neighbour.
    """

    if not clip_ids:
        raise ValueError("There are no clips to label")
    if len(clip_ids) != len(rows):
        raise ValueError("clip_ids and rows must describe the same clips")
    total = len(clip_ids)
    for offset in range(1, total):
        index = (start_index + offset) % total
        if not _is_labeled(rows[index]):
            return clip_ids[index]
    return clip_ids[(start_index + 1) % total]
