"""Pure helpers for synchronized multimodal clip playback."""

from __future__ import annotations

import math


def timeline_frame_count(modalities: dict[str, list[str]]) -> int:
    """Choose a synchronized timeline length without oversampling Thermal."""

    aligned_counts = [len(modalities.get(name, [])) for name in ("Depth_Color", "IR", "Skeleton")]
    frame_count = max(aligned_counts, default=0)
    if frame_count == 0:
        frame_count = len(modalities.get("Thermal", []))
    return max(1, frame_count)


def normalized_timeline_position(frame_index: int, frame_count: int) -> int:
    if frame_count <= 1:
        return 0
    return round(100 * frame_index / (frame_count - 1))


def playback_start_frame(frame_index: int, frame_count: int) -> int:
    """Resume within a clip, but restart when playback is already complete."""

    last_frame = max(0, frame_count - 1)
    current = max(0, min(int(frame_index), last_frame))
    return 0 if current >= last_frame else current


def playback_interval(duration_seconds: object, frame_count: int, speed: float) -> float:
    """Return a practical refresh interval that approximates recorded time."""

    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0 or frame_count <= 1:
        base_interval = 0.1
    else:
        base_interval = duration / (frame_count - 1)
    return max(0.05, min(1.0, base_interval / max(speed, 0.1)))
