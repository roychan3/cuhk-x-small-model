"""Shared vocabulary for the extractable feature blocks.

``modeling.features`` produces these blocks and the dashboard offers them for
selection, so both need the names, the default set, and the validator. This
module is deliberately standard-library only and lives beside
``visualization/progress.py`` for the same reason: ``visualization`` imports
without the modeling stack, while ``modeling`` already depends on
``visualization.dataset`` and ``visualization.progress``. Keeping one
definition here avoids a second copy in the dashboard that has to be edited in
lockstep.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Every block the extractor can produce, in the order features are laid out.
ALL_FEATURE_BLOCKS: tuple[str, ...] = (
    "depth",
    "ir",
    "imu",
    "skeleton",
    "depth_engineered",
    "ir_engineered",
    "imu_engineered",
    "skeleton_engineered",
)

#: Blocks grouped by the sensor they come from, for display.
FEATURE_BLOCK_GROUPS: dict[str, tuple[str, ...]] = {
    "Depth": ("depth", "depth_engineered"),
    "IR": ("ir", "ir_engineered"),
    "IMU": ("imu", "imu_engineered"),
    "Skeleton": ("skeleton", "skeleton_engineered"),
}

#: Everything except the legacy IR base block, which grouped validation showed
#: adds nothing over its engineered counterpart.
DEFAULT_FEATURE_BLOCKS: tuple[str, ...] = tuple(
    block for block in ALL_FEATURE_BLOCKS if block != "ir"
)

FEATURE_BLOCK_LABELS: dict[str, str] = {
    "depth": "Depth (base)",
    "depth_engineered": "Depth (engineered)",
    "ir": "IR (base)",
    "ir_engineered": "IR (engineered)",
    "imu": "IMU (base)",
    "imu_engineered": "IMU (engineered)",
    "skeleton": "Skeleton (base)",
    "skeleton_engineered": "Skeleton (engineered)",
}


def normalize_feature_blocks(blocks: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a block selection and return it in canonical order.

    The order is fixed rather than the caller's, because the result is written
    into ``validation.json`` and compared against cache contents: two requests
    for the same blocks must produce the same tuple, or the same ablation shows
    up twice in the leaderboard.
    """

    if blocks is None:
        return DEFAULT_FEATURE_BLOCKS
    requested = {str(block).strip() for block in blocks if str(block).strip()}
    unknown = sorted(requested - set(ALL_FEATURE_BLOCKS))
    if unknown:
        raise ValueError(
            f"Unknown feature block(s): {unknown}. "
            f"Available: {', '.join(ALL_FEATURE_BLOCKS)}."
        )
    if not requested:
        raise ValueError("At least one feature block must be selected")
    return tuple(block for block in ALL_FEATURE_BLOCKS if block in requested)


def describe_feature_blocks(blocks: Sequence[str]) -> str:
    """Render a block selection for captions and reports."""

    return ", ".join(normalize_feature_blocks(list(blocks)))
