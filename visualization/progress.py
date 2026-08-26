"""Shared atomic JSON progress-file writer.

One schema is used by every long-running command that publishes progress
(``modeling.train`` and ``visualization.build_manifest``) so a single reader
can follow either:

``status``
    ``running``, ``complete`` or ``error``. A reader polling only the file has
    no other way to tell a finished job from one that died mid-stage.
``stage``
    Free-form phase name within the job, e.g. ``features`` or ``writing``.
``current`` / ``total``
    Optional counters for the active stage.
``message``
    Human-readable detail.
``outputs``
    Optional mapping of produced files, present on completion.
``updated_at``
    Added automatically; lets a reader spot a stalled writer.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

RUNNING = "running"
COMPLETE = "complete"
ERROR = "error"


def write_progress(path: Path | None, **values: object) -> None:
    """Atomically replace ``path`` with a timestamped JSON progress object."""

    if path is None:
        return
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def mark_error(path: Path | None, message: str) -> None:
    """Record a failure while keeping the stage the job died in.

    Overwriting ``stage`` with a placeholder would tell a reader the job failed
    but not where, so the last published stage and counters are carried over.
    """

    if path is None:
        return
    carried: dict[str, object] = {}
    try:
        previous = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}
    if isinstance(previous, dict):
        carried = {
            key: previous[key]
            for key in ("stage", "current", "total")
            if key in previous
        }
    write_progress(path, status=ERROR, message=message, **carried)
