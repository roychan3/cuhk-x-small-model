"""Interactive Streamlit explorer for the CUHK-X small-model dataset."""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path, PurePosixPath

# Streamlit adds this script's directory to sys.path, but not necessarily the
# repository root when launched by absolute path or from another directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from visualization.algorithm_comparison import render_algorithm_comparison
from visualization.dataset import (
    MODALITIES,
    MODALITY_SLUGS,
    DataSource,
    build_clip_member_index,
    build_dataset_manifest,
    build_initial_dataset_manifest,
    build_member_index,
    discover_sources,
    read_members,
    resolve_dataset_root,
)
from visualization.manual_labels import (
    active_row_indices,
    archive_manual_label_file,
    initial_label_clip,
    load_manual_label_rows,
    manual_label_path,
    next_unlabeled_clip,
    write_manual_label_rows,
)
from visualization.playback import (
    playback_interval,
    timeline_frame_count,
    timeline_member_index,
)
from visualization.predictions import (
    PredictionTable,
    clip_id_from_submission_path,
    discover_prediction_csvs,
    load_action_mapping,
    load_prediction_csv,
)
from visualization.training_pipeline import (
    FULL_DATASET,
    render_training_pipeline,
    workflow_dataset_paths,
)

st.set_page_config(page_title="CUHK-X Dataset Explorer", page_icon="🧭", layout="wide")

INITIAL_CLIP_LIMIT = 200

#: Manual labels are recorded against the tracked test index rather than
#: whichever CSV the workflow points at. The sample dataset ships a subset of
#: these same clips, so both dataset choices label one canonical table that
#: lives in the repository; two tables could disagree about the same clip.
CANONICAL_TEST_CSV = REPOSITORY_ROOT / "Testing" / "test.csv"

SKELETON_JOINT_NAMES = (
    "Pelvis",
    "Right hip",
    "Right knee",
    "Right ankle",
    "Left hip",
    "Left knee",
    "Left ankle",
    "Spine",
    "Thorax",
    "Neck",
    "Head",
    "Left shoulder",
    "Left elbow",
    "Left wrist",
    "Right shoulder",
    "Right elbow",
    "Right wrist",
)

# Skeleton predictions use the Human3.6M 17-joint order, not COCO. Keeping the
# two orders separate is important: COCO indices turn the legs and torso into
# long crossing lines even though all of the underlying points are valid.
SKELETON_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)

SKELETON_COLORS = (
    "#00CC96",
    "#636EFA",
    "#EF553B",
    "#AB63FA",
    "#FFA15A",
)


def _skeleton_edge_chain(
    edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Order ``edges`` parent-before-child as ``(start, end, edge_index)``.

    ``_stabilize_skeleton_points`` walks outward from the pelvis and needs each
    edge's start joint to be placed before its end joint. Deriving that order
    here rather than assuming ``SKELETON_EDGES`` already has it means reordering
    the edge list cannot silently produce a scrambled skeleton — the bone
    lengths would still come out correct, so no length-based test would notice.
    """

    children: dict[int, list[tuple[int, int]]] = {}
    for index, (start, end) in enumerate(edges):
        children.setdefault(start, []).append((end, index))
    chain: list[tuple[int, int, int]] = []
    placed = {0}
    queue = [0]
    while queue:
        start = queue.pop(0)
        for end, index in children.get(start, ()):
            if end in placed:
                raise ValueError(f"SKELETON_EDGES joint {end} has more than one parent")
            placed.add(end)
            chain.append((start, end, index))
            queue.append(end)
    if len(chain) != len(edges):
        raise ValueError("SKELETON_EDGES must form one tree rooted at joint 0")
    return tuple(chain)


SKELETON_EDGE_CHAIN = _skeleton_edge_chain(SKELETON_EDGES)

SkeletonAxisRanges = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]
SkeletonBoneLengths = tuple[float, ...]


def natural_key(value: object) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", "" if value is None else str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def pretty_action(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    text = str(value)
    if "_" in text and text.split("_", 1)[0].isdigit():
        text = text.split("_", 1)[1]
    return text.replace("_", " ")


def prediction_label(action_id: object, action_name: object) -> str:
    return f"{int(action_id):02d} · {pretty_action(action_name)}"


def correctness_flag(value: object) -> bool | None:
    """Normalize a ``prediction_correct`` cell to ``True``/``False``/``None``.

    The column uses pandas' nullable ``boolean`` dtype, so scalar access yields
    ``numpy.bool_`` rather than ``bool``. Identity checks (``value is True``)
    therefore never match; convert explicitly instead.
    """

    if value is None or pd.isna(value):
        return None
    return bool(value)


def source_from_dict(data: dict[str, str]) -> DataSource:
    return DataSource(split=data["split"], kind=data["kind"], path=data["path"])


@st.cache_data(show_spinner=False)
def cached_live_manifest(
    dataset_root: str,
    deep_test: bool,
) -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    records, sources = build_dataset_manifest(dataset_root, deep_test=deep_test)
    return records, {split: source.to_dict() for split, source in sources.items()}


@st.cache_data(show_spinner=False)
def cached_initial_manifest(
    dataset_root: str,
    max_clips: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    records, sources = build_initial_dataset_manifest(dataset_root, max_clips=max_clips)
    return records, {split: source.to_dict() for split, source in sources.items()}


@st.cache_data(show_spinner=False)
def cached_saved_manifest(
    dataset_root: str,
    manifest_path: str,
    manifest_mtime_ns: int,
    manifest_size: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    # mtime and size are cache-key inputs, preventing a background rebuild at
    # the same path from leaving Streamlit's prior manifest cached indefinitely.
    del manifest_mtime_ns, manifest_size
    path = Path(manifest_path).expanduser()
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError("Saved manifest must be Parquet, CSV, or JSON.")
    for column in frame.columns:
        non_null = frame[column].dropna()
        if non_null.empty:
            continue
        normalized = {str(value).strip().lower() for value in non_null.unique()}
        if normalized <= {"true", "false"}:
            frame[column] = frame[column].map(
                lambda value: None if pd.isna(value) else str(value).strip().lower() == "true"
            )
    sources = discover_sources(dataset_root)
    return frame.where(pd.notna(frame), None).to_dict("records"), {
        split: source.to_dict() for split, source in sources.items()
    }


@st.cache_resource(show_spinner=False)
def background_manifest_process(
    dataset_root: str,
    output_path: str,
    progress_path: str,
    deep_test: bool,
) -> subprocess.Popen[str]:
    """Start one shared full-manifest builder for all Streamlit sessions."""

    command = [
        sys.executable,
        "-m",
        "visualization.build_manifest",
        "--dataset-root",
        dataset_root,
        "--output",
        output_path,
        "--progress-file",
        progress_path,
    ]
    if not deep_test:
        command.append("--no-deep-test")
    try:
        Path(progress_path).unlink(missing_ok=True)
    except OSError:
        # A read-only artifacts mount cannot be cleared. Let the builder run and
        # report the failure itself rather than taking the whole page down: this
        # call sits inside main()'s broad handler, which would st.stop().
        pass
    return subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


@st.cache_data(show_spinner=False)
def cached_member_index(source_data: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    return build_member_index(source_from_dict(source_data))


@st.cache_data(show_spinner=False)
def cached_sources(dataset_root: str) -> dict[str, dict[str, str]]:
    """Split sources for pages that skip the manifest build.

    The manifest helpers already return this alongside their records; manual
    labeling needs it without paying for a dataset scan, and it reruns on every
    button click, so the directory globs are cached here instead.
    """

    return {
        split: source.to_dict()
        for split, source in discover_sources(dataset_root).items()
    }


@st.cache_data(show_spinner=False)
def cached_clip_index(source_data: dict[str, str], clip_id: str) -> dict[str, list[str]]:
    return build_clip_member_index(source_from_dict(source_data), clip_id)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_payloads(source_data: dict[str, str], member_paths: tuple[str, ...]) -> dict[str, bytes]:
    return read_members(source_from_dict(source_data), member_paths)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_skeleton_calibration(
    source_data: dict[str, str],
    member_paths: tuple[str, ...],
) -> tuple[SkeletonAxisRanges | None, SkeletonBoneLengths | None]:
    payloads = read_members(source_from_dict(source_data), member_paths)
    return skeleton_calibration(payloads.values())


@st.cache_data(show_spinner=False)
def cached_action_mapping(
    mapping_path: str,
    mapping_mtime_ns: int,
    mapping_size: int,
) -> dict[int, str]:
    del mapping_mtime_ns, mapping_size
    return load_action_mapping(mapping_path)


@st.cache_data(show_spinner=False)
def cached_prediction_file(
    prediction_path: str,
    prediction_mtime_ns: int,
    prediction_size: int,
    action_mapping: tuple[tuple[int, str], ...],
) -> PredictionTable:
    del prediction_mtime_ns, prediction_size
    return load_prediction_csv(prediction_path, dict(action_mapping))


def modality_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modality in MODALITIES:
        column = f"{MODALITY_SLUGS[modality]}_present"
        rows.append(
            {
                "modality": modality,
                "clips": int(frame[column].fillna(False).astype(bool).sum()),
                "coverage": float(frame[column].fillna(False).astype(bool).mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def add_predictions(frame: pd.DataFrame, predictions: PredictionTable) -> pd.DataFrame:
    """Attach validated prediction IDs and names to matching clips.

    Predictions are matched by ``clip_id`` regardless of split so that both
    training and test predictions can be visualized. Training clip IDs look
    like ``<action>/<user>/<trial>`` while test IDs are ``SM_test_XXXX``,
    so a test-only CSV naturally leaves training rows as ``NA`` and a
    model-generated table that contains both splits fills both.
    """

    enriched = frame.copy()
    action_ids = {clip_id: item.action_id for clip_id, item in predictions.by_clip.items()}
    action_names = {clip_id: item.action_name for clip_id, item in predictions.by_clip.items()}
    enriched["prediction_action_id"] = enriched["clip_id"].map(action_ids).astype("Int64")
    enriched["prediction_action_name"] = enriched["clip_id"].map(action_names)
    # Convenience correctness flag for training clips (where ground truth exists).
    if "action_id" in enriched.columns:
        # Start as <NA> and fill only where a prediction exists; then mask test rows.
        enriched["prediction_correct"] = pd.NA
        has_pred = enriched["prediction_action_id"].notna() & enriched["action_id"].notna()
        # Use nullable boolean dtype
        enriched.loc[has_pred, "prediction_correct"] = (
            enriched.loc[has_pred, "prediction_action_id"].astype("Int64")
            == enriched.loc[has_pred, "action_id"].astype("Int64")
        )
        enriched["prediction_correct"] = enriched["prediction_correct"].astype("boolean")
        enriched.loc[enriched["split"] == "test", "prediction_correct"] = pd.NA
    return enriched


def render_overview(frame: pd.DataFrame) -> None:
    st.header("Dataset overview")
    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    complete = int(frame["complete"].fillna(False).astype(bool).sum())
    metric_columns = st.columns(5)
    metric_columns[0].metric("Training clips", f"{len(train):,}")
    metric_columns[1].metric("Test clips", f"{len(test):,}")
    metric_columns[2].metric("Actions", int(train["action_id"].nunique()) if not train.empty else 0)
    metric_columns[3].metric("Users", int(train["user"].nunique()) if not train.empty else 0)
    metric_columns[4].metric("All modalities", f"{complete / len(frame):.1%}" if len(frame) else "—")

    left, right = st.columns((1.45, 1))
    with left:
        st.subheader("Training class balance")
        if train.empty:
            st.info("No training source was discovered.")
        else:
            counts = (
                train.groupby(["action_id", "action_name"], dropna=False)
                .size()
                .reset_index(name="clips")
                .sort_values("action_id")
            )
            counts["action"] = counts["action_name"].map(pretty_action)
            figure = px.bar(
                counts,
                x="action",
                y="clips",
                color="clips",
                color_continuous_scale="Blues",
                hover_data=["action_id"],
            )
            figure.update_layout(xaxis_tickangle=-55, coloraxis_showscale=False, height=480)
            st.plotly_chart(figure, width="stretch")
    with right:
        st.subheader("Modality coverage")
        coverage = modality_coverage(frame)
        figure = px.bar(
            coverage,
            x="coverage",
            y="modality",
            orientation="h",
            text=coverage["coverage"].map(lambda value: f"{value:.1f}%"),
            range_x=(0, 105),
        )
        figure.update_traces(textposition="outside")
        figure.update_layout(height=480, xaxis_title="Clips with a file (%)", yaxis_title=None)
        st.plotly_chart(figure, width="stretch")

    if not train.empty:
        st.subheader("Action × user coverage")
        heat = pd.crosstab(train["user"], train["action_id"])
        heat = heat.reindex(sorted(heat.index, key=natural_key))
        figure = go.Figure(
            data=go.Heatmap(
                z=heat.values,
                x=[str(value) for value in heat.columns],
                y=list(heat.index),
                colorscale="Viridis",
                colorbar_title="clips",
            )
        )
        figure.update_layout(xaxis_title="Action ID", yaxis_title="User", height=430)
        st.plotly_chart(figure, width="stretch")

    st.subheader("Files per present clip")
    count_columns = [f"{MODALITY_SLUGS[m]}_file_count" for m in MODALITIES]
    available = [column for column in count_columns if column in frame]
    long = frame[["split", *available]].melt(id_vars="split", var_name="modality", value_name="files")
    long["modality"] = long["modality"].str.replace("_file_count", "", regex=False)
    long = long[long["files"].fillna(0) > 0]
    figure = px.box(long, x="modality", y="files", color="split", points=False, log_y=True)
    figure.update_layout(yaxis_title="Files (log scale)", xaxis_title=None, height=420)
    st.plotly_chart(figure, width="stretch")

    st.subheader("Modality combinations")
    patterns = []
    for _, row in frame.iterrows():
        present = [m for m in MODALITIES if bool(row.get(f"{MODALITY_SLUGS[m]}_present", False))]
        patterns.append(" + ".join(present) if present else "None")
    pattern_counts = pd.Series(patterns).value_counts().rename_axis("modalities").reset_index(name="clips")
    st.dataframe(pattern_counts, hide_index=True, width="stretch")


def _skeleton_people(
    data: bytes,
) -> list[tuple[list[tuple[float, float, float]], list[float]]]:
    try:
        people = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(people, list) or not people:
        return []

    parsed = []
    for person in people:
        if not isinstance(person, dict):
            continue
        keypoints = person.get("keypoints", [])
        if len(keypoints) != len(SKELETON_JOINT_NAMES):
            continue
        try:
            points = [tuple(float(value) for value in point) for point in keypoints]
        except (TypeError, ValueError):
            continue
        if any(len(point) != 3 or not all(math.isfinite(value) for value in point) for point in points):
            continue
        raw_scores = person.get("keypoint_scores", [1.0] * len(points))
        try:
            scores = [float(score) for score in raw_scores]
        except (TypeError, ValueError):
            scores = [1.0] * len(points)
        if len(scores) != len(points):
            scores = [1.0] * len(points)
        parsed.append((points, scores))
    return parsed


def _skeleton_axis_ranges_from_points(
    points: list[tuple[float, float, float]],
) -> SkeletonAxisRanges | None:
    if not points:
        return None
    bounds = tuple(
        (min(point[axis] for point in points), max(point[axis] for point in points))
        for axis in range(3)
    )
    centers = tuple((low + high) / 2.0 for low, high in bounds)
    half_span = max(max(high - low for low, high in bounds) * 0.58, 1e-3)
    return (
        (centers[0] - half_span, centers[0] + half_span),
        (centers[1] - half_span, centers[1] + half_span),
        (centers[2] - half_span, centers[2] + half_span),
    )


def _stabilize_skeleton_points(
    points: list[tuple[float, float, float]],
    bone_lengths: SkeletonBoneLengths,
) -> list[tuple[float, float, float]]:
    """Keep pose directions while enforcing clip-median bone lengths."""

    stabilized = [points[0], *([(0.0, 0.0, 0.0)] * (len(points) - 1))]
    for start, end, edge_index in SKELETON_EDGE_CHAIN:
        target_length = bone_lengths[edge_index]
        direction = tuple(points[end][axis] - points[start][axis] for axis in range(3))
        source_length = math.sqrt(sum(value * value for value in direction))
        if source_length <= 1e-8:
            stabilized[end] = stabilized[start]
            continue
        stabilized[end] = tuple(
            stabilized[start][axis] + direction[axis] * target_length / source_length
            for axis in range(3)
        )
    return stabilized


def skeleton_calibration(
    payloads: Iterable[bytes],
) -> tuple[SkeletonAxisRanges | None, SkeletonBoneLengths | None]:
    """Compute fixed display bounds and bone lengths for one clip."""

    frames = [_skeleton_people(payload) for payload in payloads]
    raw_points = [point for people in frames for points, _ in people for point in points]
    samples: list[list[float]] = [[] for _ in SKELETON_EDGES]
    for people in frames:
        for points, _ in people:
            for index, (start, end) in enumerate(SKELETON_EDGES):
                length = math.dist(points[start], points[end])
                if math.isfinite(length) and length > 1e-8:
                    samples[index].append(length)
    if not all(samples):
        return _skeleton_axis_ranges_from_points(raw_points), None

    bone_lengths = tuple(statistics.median(lengths) for lengths in samples)
    stabilized_points = [
        point
        for people in frames
        for points, _ in people
        for point in _stabilize_skeleton_points(points, bone_lengths)
    ]
    return _skeleton_axis_ranges_from_points(stabilized_points), bone_lengths


def _float_at(row: list[str], index: int) -> float | None:
    try:
        return float(row[index])
    except (IndexError, TypeError, ValueError):
        return None


def imu_figure(payloads: dict[str, bytes], paths: list[str]) -> go.Figure | None:
    points: list[dict[str, object]] = []
    for path in paths:
        text = payloads[path].decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(text)))
        for row in rows[1:]:
            ax, ay, az = (_float_at(row, index) for index in (2, 3, 4))
            gx, gy, gz = (_float_at(row, index) for index in (5, 6, 7))
            if None in (ax, ay, az, gx, gy, gz) or len(row) < 2:
                continue
            points.append(
                {
                    "time": row[0],
                    "device": row[1].split("(", 1)[0],
                    "file": PurePosixPath(path).name,
                    "acceleration": math.sqrt(ax * ax + ay * ay + az * az),
                    "gyroscope": math.sqrt(gx * gx + gy * gy + gz * gz),
                }
            )
    if not points:
        return None
    data = pd.DataFrame(points)
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["time"]).sort_values("time")
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for device, group in data.groupby("device"):
        figure.add_trace(
            go.Scattergl(x=group["time"], y=group["acceleration"], mode="lines", name=str(device)),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=group["time"],
                y=group["gyroscope"],
                mode="lines",
                name=str(device),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(title_text="|acceleration| (g)", row=1, col=1)
    figure.update_yaxes(title_text="|angular velocity| (°/s)", row=2, col=1)
    figure.update_xaxes(title_text="Time", row=2, col=1)
    figure.update_layout(height=560, hovermode="x unified", margin={"t": 30})
    return figure


PLAYER_MODALITIES = ("Depth_Color", "IR", "Thermal", "Skeleton")


def _timeline_selections(
    modalities: Mapping[str, list[str]],
    frame_count: int,
) -> list[dict[str, str | None]]:
    """Resolve every displayed modality path before client-side playback starts."""

    selections: list[dict[str, str | None]] = []
    for frame_index in range(frame_count):
        selected: dict[str, str | None] = {}
        for modality in PLAYER_MODALITIES:
            paths = modalities.get(modality, [])
            path = paths[timeline_member_index(frame_index, frame_count, len(paths))] if paths else None
            selected[modality] = path
        selections.append(selected)
    return selections


def _selected_paths(
    selections: Iterable[Mapping[str, str | None]],
    modalities: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the distinct members the player needs, in first-use order."""

    return tuple(
        dict.fromkeys(
            path
            for selection in selections
            for modality in modalities
            if (path := selection.get(modality)) is not None
        )
    )


# The player inlines every frame of a clip as a data URI, so the payload grows
# with clip length rather than with what is on screen. These steps trade
# resolution for total size once a clip is long enough that the default would
# push the whole document past a comfortable size. Depth Color and IR are
# 640x480 and Thermal is 320x240, so the first step re-encodes without
# resizing; the later ones genuinely shrink.
IMAGE_ENCODING_STEPS = ((640, 82), (512, 76), (400, 70), (320, 64))
CLIP_PLAYER_IMAGE_BUDGET = 24 * 1024 * 1024
CLIP_PLAYER_HEIGHT = 1160


def _image_encoding(image_count: int) -> tuple[int, int]:
    """Pick a resize bound and JPEG quality from how many frames must be inlined."""

    for step, (max_edge, quality) in enumerate(IMAGE_ENCODING_STEPS):
        if image_count <= 120 * (step + 1) ** 2:
            return max_edge, quality
    return IMAGE_ENCODING_STEPS[-1]


def _image_data_uri(path: str, payload: bytes, max_edge: int, quality: int) -> str | None:
    """Re-encode one frame as a JPEG data URI, or ``None`` if it cannot be shown.

    Falls back to the original bytes when Pillow cannot read the file, but only
    for suffixes a browser can actually decode; an unknown suffix returns
    ``None`` so the player reports the frame as unavailable rather than
    embedding a URI that silently fails to load.
    """

    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            if image.mode != "RGB":
                image = image.convert("RGB")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except (OSError, ValueError):
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(PurePosixPath(path).suffix.lower())
        if mime_type is None:
            return None
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"


@st.cache_data(show_spinner=False, max_entries=8)
def cached_image_assets(
    source_data: dict[str, str],
    image_paths: tuple[str, ...],
) -> tuple[dict[str, str], tuple[int, int]]:
    """Return one clip's frames as data URIs plus the encoding actually used.

    Reading through here rather than ``cached_payloads`` keeps the raw frame
    bytes transient. Caching both would hold two full copies of every clip that
    has been opened: the originals and the re-encoded JPEGs.

    The encoding is returned rather than recomputed by the caller so the size
    reported under the player cannot drift from what was encoded.
    """

    encoding = _image_encoding(len(image_paths))
    max_edge, quality = encoding
    payloads = read_members(source_from_dict(source_data), image_paths)
    assets: dict[str, str] = {}
    for path in image_paths:
        payload = payloads.get(path)
        if payload is None:
            continue
        uri = _image_data_uri(path, payload, max_edge, quality)
        if uri is not None:
            assets[path] = uri
    return assets, encoding


def _radar_playback_frames(data: bytes | None) -> list[dict[str, object]]:
    """Parse one radar CSV once, rather than rebuilding a Plotly figure per tick."""

    if not data:
        return []
    grouped: dict[int, dict[str, object]] = {}
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig", errors="replace")))

    def number(row: Mapping[str, str | None], name: str) -> float:
        try:
            return float(row.get(name) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    for row in rows:
        try:
            frame_number = int(float(row.get("frame") or ""))
        except (TypeError, ValueError):
            continue
        frame = grouped.setdefault(
            frame_number,
            {"frame": frame_number, "x": [], "y": [], "z": [], "velocity": [], "snr": []},
        )
        for column, key in (("x", "x"), ("y", "y"), ("z", "z"), ("v", "velocity"), ("snr", "snr")):
            values = frame[key]
            assert isinstance(values, list)
            values.append(number(row, column))
    return [grouped[frame_number] for frame_number in sorted(grouped)]


def _clip_player_model(
    selections: list[dict[str, str | None]],
    image_assets: Mapping[str, str],
    skeleton_payloads: Mapping[str, bytes],
    radar_payload: bytes | None,
    skeleton_ranges: SkeletonAxisRanges | None,
    skeleton_bone_lengths: SkeletonBoneLengths | None,
    playback_identity: str,
    intervals_ms: Mapping[str, int],
) -> dict[str, object]:
    # Several timeline frames map to the same skeleton file whenever the pose
    # stream is shorter than the timeline, so parse and stabilize each file once.
    poses_by_path: dict[str, list[dict[str, object]]] = {}

    def poses_for(path: str | None) -> list[dict[str, object]]:
        if path is None or path not in skeleton_payloads:
            return []
        if path not in poses_by_path:
            people = _skeleton_people(skeleton_payloads[path])
            if skeleton_bone_lengths is not None:
                people = [
                    (_stabilize_skeleton_points(points, skeleton_bone_lengths), scores)
                    for points, scores in people
                ]
            poses_by_path[path] = [
                {
                    "points": points,
                    "scores": scores,
                    "color": SKELETON_COLORS[index % len(SKELETON_COLORS)],
                    "name": f"Person {index + 1}",
                }
                for index, (points, scores) in enumerate(people)
            ]
        return poses_by_path[path]

    frames: list[dict[str, object]] = []
    for selection in selections:
        frames.append(
            {
                "images": {
                    modality: selection.get(modality)
                    for modality in ("Depth_Color", "IR", "Thermal")
                },
                "skeleton": poses_for(selection.get("Skeleton")),
            }
        )
    return {
        "identity": playback_identity,
        "frames": frames,
        "imageAssets": dict(image_assets),
        "radarFrames": _radar_playback_frames(radar_payload),
        "skeleton": {
            "edges": SKELETON_EDGES,
            "jointNames": SKELETON_JOINT_NAMES,
            "ranges": skeleton_ranges,
        },
        "intervalsMs": dict(intervals_ms),
    }


@lru_cache(maxsize=1)
def _plotly_script() -> str:
    """Escape the bundled plotly.js once; it is several megabytes and constant."""

    return get_plotlyjs().replace("</script", "<\\/script")


def _clip_player_html(model: Mapping[str, object]) -> str:
    """Return one persistent, self-contained browser player for a whole clip."""

    model_json = json.dumps(model, separators=(",", ":")).replace("</", "<\\/")
    has_radar = bool(model.get("radarFrames"))
    plotly_javascript = _plotly_script() if has_radar else ""
    template = r"""
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: transparent; font-family: sans-serif;
    color: #31333f; }
  /* The host iframe is a fixed height, but the grids below collapse to one
     column on narrow viewports and the image panels grow with width. Scroll
     inside the player so neither end of that range clips content: st.iframe
     has no scrolling parameter to fall back on. */
  #player { height: 100%; overflow-y: auto; overflow-x: hidden; padding-right: 4px; }
  button, select, input { font: inherit; }
  .toolbar { display: grid; grid-template-columns: auto auto minmax(125px, auto) 1fr;
    gap: 10px; align-items: end; margin: 0 0 12px; }
  button, select { min-height: 38px; border: 1px solid #c8cdd4; border-radius: 7px;
    background: #fff; color: inherit; padding: 7px 12px; }
  button { cursor: pointer; font-weight: 600; }
  button:disabled { cursor: wait; opacity: .55; }
  label { display: grid; gap: 3px; font-size: 12px; color: #60646c; }
  #timeline { width: 100%; margin-bottom: 5px; }
  #timelineText { font-size: 13px; color: #60646c; white-space: nowrap; }
  #loadStatus { min-height: 18px; margin-bottom: 4px; font-size: 12px; color: #60646c; }
  h3 { margin: 12px 0 8px; font-size: 1.25rem; }
  h4 { margin: 0 0 8px; font-size: 1rem; }
  .stream-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
  .lower-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px;
    margin-top: 18px; }
  .panel { min-width: 0; }
  .frame { position: relative; width: 100%; overflow: hidden; border: 1px solid #d8dce2;
    border-radius: 7px; background: #090b0f; }
  .image-frame { aspect-ratio: 4 / 3; }
  .viewer-frame { height: 520px; }
  canvas { display: block; width: 100%; height: 100%; }
  #skeleton { cursor: grab; touch-action: none; }
  #skeleton.dragging { cursor: grabbing; }
  .caption { min-height: 34px; padding-top: 5px; color: #60646c; font-size: 12px;
    overflow-wrap: anywhere; }
  .hint { position: absolute; left: 10px; top: 8px; color: #8a919d; font-size: 12px;
    pointer-events: none; }
  #resetCamera { position: absolute; right: 10px; top: 8px; z-index: 2; min-height: 30px;
    padding: 4px 9px; background: rgba(255,255,255,.88); color: #31333f; }
  #tooltip { display: none; position: absolute; pointer-events: none; z-index: 3;
    padding: 5px 7px; border-radius: 5px; background: rgba(30,34,40,.92); color: white;
    font-size: 12px; }
  #radar { width: 100%; height: 518px; }
  .message { display: grid; place-items: center; width: 100%; height: 100%; color: #8a919d;
    font-size: 14px; }
  @media (prefers-color-scheme: dark) {
    html, body { color: #fafafa; }
    button, select { background: #1c1f26; border-color: #4b515c; }
    label, #timelineText, #loadStatus, .caption { color: #a9afb9; }
    .frame { border-color: #3d434d; }
  }
  @media (max-width: 760px) {
    .toolbar { grid-template-columns: 1fr 1fr; }
    .toolbar .timeline-control { grid-column: 1 / -1; }
    .stream-grid, .lower-grid { grid-template-columns: 1fr; }
  }
</style>
<div id="player">
  <div class="toolbar">
    <button id="toggle" type="button" disabled>▶ Play</button>
    <button id="restart" type="button">↺ Restart</button>
    <label>Speed
      <select id="speed">
        <option value="0.5">0.5×</option>
        <option value="1" selected>1×</option>
        <option value="2">2×</option>
      </select>
    </label>
    <div class="timeline-control">
      <input id="timeline" type="range" min="0" value="0" step="1">
      <div id="timelineText"></div>
    </div>
  </div>
  <div id="loadStatus">Loading clip frames…</div>
  <h3>Visual streams</h3>
  <div class="stream-grid">
    <div class="panel"><h4>Depth_Color</h4><div class="frame image-frame"><canvas id="image-Depth_Color"></canvas></div><div class="caption" id="caption-Depth_Color"></div></div>
    <div class="panel"><h4>IR</h4><div class="frame image-frame"><canvas id="image-IR"></canvas></div><div class="caption" id="caption-IR"></div></div>
    <div class="panel"><h4>Thermal</h4><div class="frame image-frame"><canvas id="image-Thermal"></canvas></div><div class="caption" id="caption-Thermal"></div></div>
  </div>
  <div class="lower-grid">
    <div class="panel">
      <h3>3D skeleton</h3>
      <div class="frame viewer-frame" id="skeletonFrame">
        <canvas id="skeleton"></canvas>
        <div class="hint">Drag to rotate · Scroll to zoom</div>
        <button id="resetCamera" type="button">Reset view</button>
        <div id="tooltip"></div>
      </div>
      <div class="caption">Human3.6M 17-joint pose · camera-aligned · clip-stabilized limb lengths.</div>
    </div>
    <div class="panel">
      <h3>Radar point cloud</h3>
      <div class="frame viewer-frame"><div id="radar"></div></div>
      <div class="caption" id="radarCaption"></div>
    </div>
  </div>
</div>
<script>__PLOTLY__</script>
<script>
(() => {
  const model = __MODEL__;
  const modalities = ["Depth_Color", "IR", "Thermal"];
  const frameCount = Math.max(1, model.frames.length);
  const toggle = document.getElementById("toggle");
  const restart = document.getElementById("restart");
  const speedControl = document.getElementById("speed");
  const timeline = document.getElementById("timeline");
  const timelineText = document.getElementById("timelineText");
  const loadStatus = document.getElementById("loadStatus");
  const skeletonCanvas = document.getElementById("skeleton");
  const tooltip = document.getElementById("tooltip");
  const radar = document.getElementById("radar");
  const radarCaption = document.getElementById("radarCaption");
  timeline.max = String(frameCount - 1);

  let currentFrame = 0;
  let playing = false;
  let timer = null;
  let timerGeneration = 0;
  let ready = false;
  let radarReady = false;
  let radarInitializing = false;
  const imageCache = new Map();
  const imagePromises = new Map();

  // Every async path below ends in a promise nothing awaits. Without this the
  // failure mode is a silent deadlock: the status line keeps saying "Loading",
  // or playback stops while the button still reads "Pause", with no clue why.
  function reportError(stage, error) {
    console.error(`clip player: ${stage}`, error);
    loadStatus.textContent = `${stage} failed: ${error && error.message ? error.message : error}`;
    return null;
  }

  function canvasSize(canvas) {
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const ratio = window.devicePixelRatio || 1;
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return {context, width, height};
  }

  function message(context, width, height, value) {
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#8a919d";
    context.font = "14px sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(value, width / 2, height / 2);
  }

  function imagePathsForFrame(index) {
    if (index < 0 || index >= frameCount) return [];
    return modalities
      .map(modality => model.frames[index]?.images?.[modality] || null)
      .filter(path => path !== null && model.imageAssets[path]);
  }

  function ensureImage(path) {
    if (imagePromises.has(path)) return imagePromises.get(path);
    const promise = new Promise(resolve => {
      const image = new Image();
      image.onload = () => resolve();
      image.onerror = () => resolve();
      imageCache.set(path, image);
      image.src = model.imageAssets[path];
    });
    imagePromises.set(path, promise);
    return promise;
  }

  function ensureFrame(index) {
    return Promise.all(imagePathsForFrame(index).map(ensureImage));
  }

  function warmFrames(first, last) {
    const promises = [];
    for (let index = Math.max(0, first); index <= Math.min(frameCount - 1, last); index++) {
      promises.push(ensureFrame(index));
    }
    return Promise.all(promises);
  }

  function trimImageCache(center) {
    const retained = new Set();
    for (let index = Math.max(0, center - 2); index <= Math.min(frameCount - 1, center + 5); index++) {
      imagePathsForFrame(index).forEach(path => retained.add(path));
    }
    for (const path of imageCache.keys()) {
      if (!retained.has(path)) {
        // Dropping the reference is enough to release the decoded bitmap.
        // Assigning src = "" would re-request the document URL and fire error.
        imageCache.delete(path);
        imagePromises.delete(path);
      }
    }
  }

  const warmAround = frame =>
    warmFrames(frame + 1, frame + 4)
      .then(() => trimImageCache(currentFrame))
      .catch(error => reportError("Preloading frames", error));

  function drawImage(modality) {
    const canvas = document.getElementById(`image-${modality}`);
    const {context, width, height} = canvasSize(canvas);
    const path = model.frames[currentFrame]?.images?.[modality] || null;
    const caption = document.getElementById(`caption-${modality}`);
    caption.textContent = path ? path.split("/").pop() : "Unavailable";
    if (!path) {
      message(context, width, height, "Unavailable");
      return;
    }
    const image = imageCache.get(path);
    if (!ready || !image || !image.complete || !image.naturalWidth) {
      message(context, width, height, ready ? "Could not decode frame" : "Loading…");
      return;
    }
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#090b0f";
    context.fillRect(0, 0, width, height);
    const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
    const drawWidth = image.naturalWidth * scale;
    const drawHeight = image.naturalHeight * scale;
    context.drawImage(image, (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight);
  }

  const cameraStorageKey = `cuhkx:skeleton-camera:${model.identity}`;
  const defaultCamera = {yaw: 0, pitch: 0, zoom: 1};
  const clampCamera = value => {
    const number = (input, fallback) => Number.isFinite(input) ? input : fallback;
    return {
      yaw: number(value.yaw, 0),
      pitch: Math.max(-1.45, Math.min(1.45, number(value.pitch, 0))),
      zoom: Math.max(0.55, Math.min(2.5, number(value.zoom, 1))),
    };
  };
  let camera = {...defaultCamera};
  try {
    const stored = JSON.parse(window.parent.sessionStorage.getItem(cameraStorageKey) || "{}");
    if (stored && typeof stored === "object") camera = clampCamera({...camera, ...stored});
  } catch (_) {}
  let dragging = false;
  let lastPointer = null;
  let projectedJoints = [];
  const scaleVector = (value, amount) => value.map(item => item * amount);
  const dot = (a, b) => a.reduce((total, value, index) => total + value * b[index], 0);
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const normalize = value => {
    const length = Math.sqrt(dot(value, value)) || 1;
    return scaleVector(value, 1 / length);
  };

  function saveCamera() {
    try { window.parent.sessionStorage.setItem(cameraStorageKey, JSON.stringify(camera)); } catch (_) {}
  }

  function cameraBasis() {
    const horizontal = Math.cos(camera.pitch);
    const eye = normalize([
      Math.sin(camera.yaw) * horizontal,
      Math.cos(camera.yaw) * horizontal,
      Math.sin(camera.pitch),
    ]);
    const forward = scaleVector(eye, -1);
    const right = normalize(cross(forward, [0, 0, 1]));
    return {eye, right, up: normalize(cross(right, forward))};
  }

  function drawSkeleton() {
    const {context, width, height} = canvasSize(skeletonCanvas);
    context.clearRect(0, 0, width, height);
    const ranges = model.skeleton.ranges;
    const people = model.frames[currentFrame]?.skeleton || [];
    if (!ranges || !people.length) {
      message(context, width, height, ranges ? "No valid person pose in this frame" : "Skeleton unavailable");
      projectedJoints = [];
      return;
    }
    const centers = ranges.map(range => (range[0] + range[1]) / 2);
    const halfSpan = Math.max((ranges[0][1] - ranges[0][0]) / 2, 1e-6);
    const basis = cameraBasis();
    const project = point => {
      const centered = point.map((value, index) => (value - centers[index]) / halfSpan);
      const pixels = Math.min(width, height) * 0.37 * camera.zoom;
      return {
        x: width / 2 + dot(centered, basis.right) * pixels,
        y: height / 2 - dot(centered, basis.up) * pixels,
        depth: dot(centered, basis.eye),
      };
    };

    const cube = [];
    for (let index = 0; index < 8; index++) {
      cube.push(project([
        centers[0] + (index & 1 ? halfSpan : -halfSpan),
        centers[1] + (index & 2 ? halfSpan : -halfSpan),
        centers[2] + (index & 4 ? halfSpan : -halfSpan),
      ]));
    }
    context.strokeStyle = "rgba(128,136,148,.28)";
    context.lineWidth = 1;
    for (const [start, end] of [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]]) {
      context.beginPath();
      context.moveTo(cube[start].x, cube[start].y);
      context.lineTo(cube[end].x, cube[end].y);
      context.stroke();
    }

    projectedJoints = [];
    for (const person of people) {
      const projected = person.points.map(project);
      const bones = model.skeleton.edges.map(edge => ({
        edge,
        depth: (projected[edge[0]].depth + projected[edge[1]].depth) / 2,
      })).sort((a, b) => a.depth - b.depth);
      context.strokeStyle = person.color;
      context.lineWidth = 6;
      context.lineCap = "round";
      for (const bone of bones) {
        const start = projected[bone.edge[0]];
        const end = projected[bone.edge[1]];
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
      }
      projected.forEach((point, index) => {
        const radius = Math.max(4, Math.min(8, person.scores[index] * 7));
        context.beginPath();
        context.arc(point.x, point.y, radius, 0, Math.PI * 2);
        context.fillStyle = person.color;
        context.fill();
        context.strokeStyle = "white";
        context.lineWidth = 1;
        context.stroke();
        projectedJoints.push({
          ...point,
          radius,
          label: `${person.name} · ${model.skeleton.jointNames[index]} · score=${person.scores[index].toFixed(3)}`,
        });
      });
    }
  }

  function radarAxis(values) {
    if (!values.length) return [-1, 1];
    const low = Math.min(...values);
    const high = Math.max(...values);
    const padding = Math.max((high - low) * 0.08, 0.05);
    return [low - padding, high + padding];
  }
  const allRadar = model.radarFrames.flatMap(frame => frame.x.map((_, index) => ({
    x: frame.x[index], y: frame.y[index], z: frame.z[index], velocity: frame.velocity[index],
  })));
  const radarRanges = {
    x: radarAxis(allRadar.map(point => point.x)),
    y: radarAxis(allRadar.map(point => point.y)),
    z: radarAxis(allRadar.map(point => point.z)),
  };
  const maxVelocity = Math.max(0.001, ...allRadar.map(point => Math.abs(point.velocity)));

  function radarFrameFor(index) {
    if (!model.radarFrames.length) return null;
    const radarIndex = frameCount <= 1 ? 0 : Math.round((model.radarFrames.length - 1) * index / (frameCount - 1));
    return model.radarFrames[radarIndex];
  }

  function radarTrace(frame) {
    return {
      type: "scatter3d",
      mode: "markers",
      x: frame.x,
      y: frame.y,
      z: frame.z,
      marker: {
        size: frame.snr.map(value => Math.max(3, Math.min(11, value / 30))),
        color: frame.velocity,
        colorscale: "RdBu",
        cmin: -maxVelocity,
        cmax: maxVelocity,
        cmid: 0,
        colorbar: {title: "velocity"},
        opacity: 0.82,
      },
      text: frame.snr.map((value, index) => `frame=${frame.frame}<br>SNR=${value.toFixed(0)}<br>v=${frame.velocity[index].toFixed(3)}`),
      hoverinfo: "text",
    };
  }

  function updateRadar() {
    const frame = radarFrameFor(currentFrame);
    if (!frame || typeof Plotly === "undefined") {
      if (radar.dataset.empty !== "true") {
        radar.innerHTML = '<div class="message">Radar unavailable or contains no detections</div>';
        radar.dataset.empty = "true";
        radarCaption.textContent = "";
      }
      return;
    }
    radarCaption.textContent = `Radar frame ${frame.frame}; ${model.radarFrames.length} frames contain detections.`;
    if (!radarReady) {
      if (radarInitializing) return;
      radarInitializing = true;
      const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      Plotly.newPlot(radar, [radarTrace(frame)], {
        height: 518,
        margin: {l: 0, r: 0, t: 26, b: 0},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {color: dark ? "#fafafa" : "#31333f"},
        uirevision: model.identity,
        scene: {
          aspectmode: "data",
          uirevision: model.identity,
          xaxis: {title: "x", range: radarRanges.x},
          yaxis: {title: "y", range: radarRanges.y},
          zaxis: {title: "z", range: radarRanges.z},
        },
      }, {responsive: true, displaylogo: false}).then(() => {
        radarReady = true;
        radarInitializing = false;
        updateRadar();
      }).catch(error => {
        // Clear the guard, or every later updateRadar() returns early and the
        // panel stays dead for the rest of the session.
        radarInitializing = false;
        radar.innerHTML = '<div class="message">Radar failed to render</div>';
        radar.dataset.empty = "true";
        radarCaption.textContent = "";
        console.error("clip player: radar", error);
      });
      return;
    }
    const trace = radarTrace(frame);
    Plotly.restyle(radar, {
      x: [trace.x], y: [trace.y], z: [trace.z], text: [trace.text],
      "marker.size": [trace.marker.size], "marker.color": [trace.marker.color],
    }, [0]);
  }

  function updateControls() {
    timeline.value = String(currentFrame);
    const position = frameCount <= 1 ? 0 : Math.round(100 * currentFrame / (frameCount - 1));
    timelineText.textContent = `Frame ${currentFrame + 1} of ${frameCount} · ${position}% through clip · ${Number(speedControl.value)}×`;
    toggle.textContent = playing ? "⏸ Pause" : (currentFrame >= frameCount - 1 && frameCount > 1 ? "↻ Replay" : "▶ Play");
  }

  function drawFrame(index) {
    currentFrame = Math.max(0, Math.min(Number(index) || 0, frameCount - 1));
    modalities.forEach(drawImage);
    drawSkeleton();
    updateRadar();
    updateControls();
  }

  function intervalMs() {
    return model.intervalsMs[speedControl.value] || model.intervalsMs["1"] || 100;
  }

  function stopTimer() {
    timerGeneration += 1;
    if (timer !== null) window.clearTimeout(timer);
    timer = null;
  }

  function schedule() {
    stopTimer();
    if (!playing) return;
    const generation = timerGeneration;
    timer = window.setTimeout(async () => {
      try {
        if (currentFrame >= frameCount - 1) {
          playing = false;
          updateControls();
          return;
        }
        const nextFrame = currentFrame + 1;
        await ensureFrame(nextFrame);
        if (!playing || generation !== timerGeneration) return;
        drawFrame(nextFrame);
        warmAround(nextFrame);
        if (currentFrame >= frameCount - 1) playing = false;
        updateControls();
        schedule();
      } catch (error) {
        // Leave the controls consistent with reality rather than showing
        // "Pause" over a timer chain that has already ended.
        playing = false;
        stopTimer();
        updateControls();
        reportError("Playback", error);
      }
    }, intervalMs());
  }

  // An async listener that rejects is an unhandled rejection the user never
  // sees, so route every one of them through the same reporting path.
  const guard = (stage, handler) => async event => {
    try {
      await handler(event);
    } catch (error) {
      playing = false;
      stopTimer();
      updateControls();
      reportError(stage, error);
    }
  };
  toggle.addEventListener("click", guard("Playback", async () => {
    if (!ready || frameCount <= 1) return;
    if (playing) {
      playing = false;
      stopTimer();
    } else {
      if (currentFrame >= frameCount - 1) {
        await ensureFrame(0);
        drawFrame(0);
      }
      playing = true;
      schedule();
    }
    updateControls();
  }));
  restart.addEventListener("click", guard("Restart", async () => {
    playing = false;
    stopTimer();
    await ensureFrame(0);
    drawFrame(0);
    warmAround(0);
  }));
  speedControl.addEventListener("change", () => {
    updateControls();
    schedule();
  });
  timeline.addEventListener("input", guard("Seeking", async event => {
    stopTimer();
    const requestedFrame = Number(event.target.value);
    await ensureFrame(requestedFrame);
    if (Number(timeline.value) !== requestedFrame) return;
    drawFrame(requestedFrame);
    warmAround(requestedFrame);
    schedule();
  }));

  skeletonCanvas.addEventListener("pointerdown", event => {
    dragging = true;
    lastPointer = [event.clientX, event.clientY];
    skeletonCanvas.classList.add("dragging");
    skeletonCanvas.setPointerCapture(event.pointerId);
    tooltip.style.display = "none";
  });
  skeletonCanvas.addEventListener("pointermove", event => {
    if (dragging) {
      const dx = event.clientX - lastPointer[0];
      const dy = event.clientY - lastPointer[1];
      camera.yaw -= dx * 0.008;
      camera.pitch = Math.max(-1.45, Math.min(1.45, camera.pitch + dy * 0.008));
      lastPointer = [event.clientX, event.clientY];
      saveCamera();
      drawSkeleton();
      return;
    }
    const bounds = skeletonCanvas.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    const nearest = projectedJoints
      .map(joint => ({joint, distance: Math.hypot(joint.x - x, joint.y - y)}))
      .sort((a, b) => a.distance - b.distance)[0];
    if (!nearest || nearest.distance > nearest.joint.radius + 5) {
      tooltip.style.display = "none";
      return;
    }
    tooltip.textContent = nearest.joint.label;
    tooltip.style.left = `${x + 12}px`;
    tooltip.style.top = `${y + 12}px`;
    tooltip.style.display = "block";
  });
  skeletonCanvas.addEventListener("pointerup", event => {
    dragging = false;
    skeletonCanvas.classList.remove("dragging");
    skeletonCanvas.releasePointerCapture(event.pointerId);
    saveCamera();
  });
  skeletonCanvas.addEventListener("pointercancel", () => {
    dragging = false;
    skeletonCanvas.classList.remove("dragging");
  });
  skeletonCanvas.addEventListener("wheel", event => {
    event.preventDefault();
    camera.zoom = Math.max(0.55, Math.min(2.5, camera.zoom * Math.exp(-event.deltaY * 0.001)));
    saveCamera();
    drawSkeleton();
  }, {passive: false});
  document.getElementById("resetCamera").addEventListener("click", () => {
    camera = {...defaultCamera};
    saveCamera();
    drawSkeleton();
  });

  new ResizeObserver(() => {
    modalities.forEach(drawImage);
    drawSkeleton();
  }).observe(document.getElementById("player"));

  drawFrame(0);
  warmFrames(0, 4).then(() => {
    ready = true;
    toggle.disabled = frameCount <= 1;
    const frameLabel = frameCount === 1 ? "1 synchronized frame" : `${frameCount} synchronized frames`;
    loadStatus.textContent = Object.keys(model.imageAssets).length
      ? `${frameLabel} ready`
      : "No image streams available";
    drawFrame(currentFrame);
    trimImageCache(currentFrame);
  }).catch(error => {
    // This is the only path that enables Play; without a catch a failure here
    // leaves the player disabled under a permanent "Loading clip frames…".
    reportError("Loading clip frames", error);
  });
})();
</script>
"""
    return template.replace("__MODEL__", model_json).replace("__PLOTLY__", plotly_javascript)


def render_clip_player(
    modalities: dict[str, list[str]],
    source_data: dict[str, str],
    static_payloads: Mapping[str, bytes],
    skeleton_ranges: SkeletonAxisRanges | None,
    skeleton_bone_lengths: SkeletonBoneLengths | None,
    playback_identity: str,
    duration_seconds: object,
) -> None:
    with st.spinner("Preparing smooth playback…"):
        frame_count = timeline_frame_count(modalities)
        selections = _timeline_selections(modalities, frame_count)
        image_paths = _selected_paths(selections, ("Depth_Color", "IR", "Thermal"))
        skeleton_paths = _selected_paths(selections, ("Skeleton",))
        if image_paths:
            image_assets, (max_edge, quality) = cached_image_assets(source_data, image_paths)
        else:
            image_assets, (max_edge, quality) = {}, _image_encoding(0)
        skeleton_payloads = cached_payloads(source_data, skeleton_paths) if skeleton_paths else {}
        radar_paths = modalities.get("Radar", [])
        radar_payload = static_payloads.get(radar_paths[0]) if radar_paths else None
        intervals_ms = {
            f"{speed:g}": round(playback_interval(duration_seconds, frame_count, speed) * 1000)
            for speed in (0.5, 1.0, 2.0)
        }
        model = _clip_player_model(
            selections,
            image_assets,
            skeleton_payloads,
            radar_payload,
            skeleton_ranges,
            skeleton_bone_lengths,
            playback_identity,
            intervals_ms,
        )
        player_html = _clip_player_html(model)

    # The whole clip travels to the browser in one document, so make its size
    # visible rather than letting a long clip quietly stall the connection.
    if hasattr(st, "iframe"):
        st.iframe(player_html, width="stretch", height=CLIP_PLAYER_HEIGHT, tab_index=0)
    else:  # pragma: no cover - compatibility with Streamlit before 1.62
        components.html(player_html, height=CLIP_PLAYER_HEIGHT, scrolling=True)
    # The document is almost entirely base64, so its character count is within a
    # rounding error of its byte count and avoids copying several megabytes.
    payload_bytes = len(player_html)
    frame_label = "frame" if frame_count == 1 else "frames"
    st.caption(
        f"{frame_count:,} {frame_label} preloaded · {len(image_assets):,} images at "
        f"{max_edge}px/q{quality} · {payload_bytes / 1_048_576:.1f} MiB sent to the browser"
    )
    if payload_bytes > CLIP_PLAYER_IMAGE_BUDGET:
        st.warning(
            f"This clip's player is {payload_bytes / 1_048_576:.0f} MiB. Long clips can be slow "
            "to load or exceed the Streamlit websocket message limit "
            "(`server.maxMessageSize`)."
        )


NO_PREDICTION_OPTION = ""


def prediction_file_options(
    candidates: Iterable[Path],
    handoff: str,
    repository_root: Path,
) -> tuple[list[str], str]:
    """Return the picker's options and the value a fresh hand-off should select.

    ``handoff`` is ``prediction_csv_input``, written by the Workflow page. It is
    normalized to an absolute path so it can be compared with the discovered
    candidates, and included as an option even when it lives outside
    ``outputs/`` — discovery only scans that folder, but Workflow may write
    anywhere in the repository.
    """

    options = [NO_PREDICTION_OPTION, *(str(path) for path in candidates)]
    resolved = ""
    if handoff.strip():
        path = Path(handoff.strip()).expanduser()
        if not path.is_absolute():
            path = repository_root / path
        resolved = str(path)
        if resolved not in options and path.is_file():
            options.insert(1, resolved)
    if resolved in options:
        return options, resolved
    return options, options[1] if len(options) > 1 else NO_PREDICTION_OPTION


def _select_prediction_file(candidates: list[Path]) -> str:
    """Render the explorer's prediction picker and return the chosen path.

    The choice is deliberately local to the clip explorer, but a prediction the
    user just produced on Workflow should still win. Re-applying the hand-off on
    every render would fight the dropdown, and applying it only when the key is
    unset would ignore every later hand-off, so it is applied whenever the
    hand-off value itself changes.
    """

    handoff = str(st.session_state.get("prediction_csv_input", "")).strip()
    options, default = prediction_file_options(candidates, handoff, REPOSITORY_ROOT)

    if st.session_state.get("prediction_handoff_applied") != handoff:
        st.session_state["prediction_handoff_applied"] = handoff
        st.session_state["clip_explorer_prediction_file"] = default
    # A previously chosen file may since have been deleted or renamed.
    if st.session_state.setdefault("clip_explorer_prediction_file", default) not in options:
        st.session_state["clip_explorer_prediction_file"] = default

    selected = st.selectbox(
        "Prediction file",
        options,
        format_func=lambda value: Path(value).name if value else "(No predictions)",
        key="clip_explorer_prediction_file",
        help="Prediction CSVs discovered under outputs/, newest first. This choice affects only the clip explorer.",
    )
    if selected:
        st.caption(f"Selected predictions: `{selected}`")
    elif candidates:
        st.caption("No prediction file selected — the explorer shows ground truth only.")
    else:
        st.caption("No compatible prediction CSV files found in `outputs/`.")
    return str(selected)


def _apply_predictions(frame: pd.DataFrame, prediction_path: Path) -> pd.DataFrame:
    """Attach one prediction CSV to a copy of the frame and report its coverage."""

    dataset_hint = str(st.session_state.get("dataset_root_input", "")).strip()
    mapping_candidates = [REPOSITORY_ROOT / "Training" / "class_mapping.csv"]
    if dataset_hint:
        mapping_candidates.append(Path(dataset_hint).expanduser() / "Training" / "class_mapping.csv")
    mapping_path = next((path for path in mapping_candidates if path.is_file()), None)
    if mapping_path is None:
        raise FileNotFoundError(
            "Training/class_mapping.csv was not found in the repository or dataset root"
        )
    mapping_stat = mapping_path.stat()
    action_mapping = cached_action_mapping(
        str(mapping_path), mapping_stat.st_mtime_ns, mapping_stat.st_size
    )
    prediction_stat = prediction_path.stat()
    table = cached_prediction_file(
        str(prediction_path),
        prediction_stat.st_mtime_ns,
        prediction_stat.st_size,
        tuple(sorted(action_mapping.items())),
    )

    enriched = add_predictions(frame, table)
    clip_ids = {
        split: set(enriched.loc[enriched["split"] == split, "clip_id"].astype(str))
        for split in ("train", "test")
    }
    matched = {
        split: len(ids & set(table.by_clip)) for split, ids in clip_ids.items()
    }
    parts = [f"{count:,} {split}" for split, count in matched.items() if count]
    summary = " and ".join(parts) if parts else "no visible"
    st.caption(
        f"Loaded {len(table.by_clip):,} predictions from `{prediction_path}` · "
        f"matched {summary} clips in the current view."
    )
    if table.blank_predictions:
        st.warning(
            f"Ignored {table.blank_predictions:,} row(s) with blank predictions."
        )
    return enriched


def render_clip_explorer(
    frame: pd.DataFrame,
    sources: dict[str, dict[str, str]],
    prediction_candidates: list[Path],
) -> None:
    st.header("Multimodal clip explorer")

    # Prediction choice is local to this page, so Overview and Data quality are
    # unaffected by it.
    prediction_csv = _select_prediction_file(prediction_candidates)
    # Enrich a local copy so Overview and Data quality stay prediction-free.
    explorer_frame = frame
    if prediction_csv:
        try:
            explorer_frame = _apply_predictions(frame, Path(prediction_csv))
        except Exception as exc:
            st.error(f"Could not load predictions: {exc}")
    # From here on the explorer works on the enriched local copy.
    frame = explorer_frame
    splits = [split for split in ("test", "train") if split in set(frame["split"]) and split in sources]
    if not splits:
        st.warning("No indexed split is available.")
        return
    split = st.selectbox("Split", splits)
    filtered = frame[frame["split"] == split].copy()

    has_predictions = "prediction_action_id" in filtered.columns and not filtered.dropna(
        subset=["prediction_action_id"]
    ).empty

    if split == "train" and not filtered.empty:
        actions = (
            filtered[["action_id", "action_name"]]
            .drop_duplicates()
            .sort_values("action_id")
            .to_dict("records")
        )
        action_labels = {
            f"{int(item['action_id']):02d} · {pretty_action(item['action_name'])}": int(item["action_id"])
            for item in actions
        }
        action_label = st.selectbox("Action", list(action_labels))
        filtered = filtered[filtered["action_id"].fillna(-1).astype(int) == action_labels[action_label]]
        users = sorted(filtered["user"].dropna().unique(), key=natural_key)
        user = st.selectbox("User", users)
        filtered = filtered[filtered["user"] == user]

        if has_predictions:
            # After narrowing to action/user, offer predicted-action and correctness filters.
            pred_filtered = filtered.dropna(subset=["prediction_action_id"])
            if not pred_filtered.empty:
                prediction_options = {
                    prediction_label(action_id, group["prediction_action_name"].iloc[0]): int(action_id)
                    for action_id, group in pred_filtered.groupby("prediction_action_id")
                }
                selected_prediction = st.selectbox(
                    "Predicted action (train)",
                    ["All training clips", *prediction_options],
                    key="train_predicted_action",
                )
                if selected_prediction != "All training clips":
                    filtered = filtered[
                        filtered["prediction_action_id"].fillna(-1).astype(int)
                        == prediction_options[selected_prediction]
                    ]
                correctness = st.selectbox(
                    "Correctness",
                    ["All", "Correct only", "Incorrect only"],
                    key="train_correctness",
                )
                if correctness == "Correct only":
                    filtered = filtered[filtered["prediction_correct"] == True]  # noqa: E712
                elif correctness == "Incorrect only":
                    filtered = filtered[filtered["prediction_correct"] == False]  # noqa: E712
    elif split == "test" and has_predictions:
        predicted = filtered.dropna(subset=["prediction_action_id"])
        if not predicted.empty:
            prediction_options = {
                prediction_label(action_id, group["prediction_action_name"].iloc[0]): int(action_id)
                for action_id, group in predicted.groupby("prediction_action_id")
            }
            selected_prediction = st.selectbox(
                "Predicted action",
                ["All test clips", *prediction_options],
            )
            if selected_prediction != "All test clips":
                filtered = filtered[
                    filtered["prediction_action_id"].fillna(-1).astype(int)
                    == prediction_options[selected_prediction]
                ]

    clip_ids = sorted(filtered["clip_id"].dropna().unique(), key=natural_key)
    if not clip_ids:
        st.info("No clips match the filters.")
        return

    def clip_label(clip_id: str) -> str:
        if "prediction_action_id" not in filtered.columns:
            return clip_id
        matches = filtered[filtered["clip_id"] == clip_id]
        if matches.empty or pd.isna(matches.iloc[0].get("prediction_action_id")):
            return f"{clip_id} · no prediction"
        item = matches.iloc[0]
        pred_label = prediction_label(item["prediction_action_id"], item["prediction_action_name"])
        if split == "train":
            true_label = prediction_label(item["action_id"], item["action_name"])
            correct = correctness_flag(item.get("prediction_correct"))
            mark = "✓" if correct else ("✗" if correct is False else "")
            return f"{clip_id} · true {true_label} · pred {pred_label} {mark}".strip()
        return f"{clip_id} · {pred_label}"

    clip_id = st.selectbox("Clip", clip_ids, format_func=clip_label)
    row = filtered[filtered["clip_id"] == clip_id].iloc[0]

    if "prediction_action_id" in row.index and pd.notna(row.get("prediction_action_id")):
        if split == "train":
            true_label = prediction_label(row["action_id"], row["action_name"])
            pred_label = prediction_label(row["prediction_action_id"], row["prediction_action_name"])
            correct = correctness_flag(row.get("prediction_correct"))
            if correct:
                st.success(f"Predicted: {pred_label} — correct (true: {true_label}) ✓")
            elif correct is False:
                st.error(f"Predicted: {pred_label} — incorrect (true: {true_label}) ✗")
            else:
                st.info(f"Predicted: {pred_label} (true: {true_label})")
        else:
            st.success(
                "Predicted action: "
                f"{prediction_label(row['prediction_action_id'], row['prediction_action_name'])}"
            )
    elif "prediction_action_id" in row.index:
        st.warning("This clip has no prediction in the selected source.")

    counts = st.columns(6)
    for column, modality in zip(counts, MODALITIES):
        column.metric(modality, int(row.get(f"{MODALITY_SLUGS[modality]}_file_count", 0) or 0))
    if row.get("issues"):
        st.warning(str(row["issues"]))

    source_data = sources[split]
    source = source_from_dict(source_data)
    if not source.readable:
        st.info(
            "Training metadata is available, but frame playback is disabled for the multipart ZIP. "
            "Merge it into HAR_full.zip or extract HAR/ to enable this page."
        )
        return

    with st.spinner("Indexing clip…"):
        modalities = cached_clip_index(source_data, clip_id)
    imu_paths = modalities.get("IMU", [])
    radar_paths = modalities.get("Radar", [])
    skeleton_paths = modalities.get("Skeleton", [])
    static_paths = tuple(dict.fromkeys([*imu_paths, *radar_paths]))
    static_payloads = cached_payloads(source_data, static_paths) if static_paths else {}
    if skeleton_paths:
        skeleton_ranges, skeleton_bone_lengths = cached_skeleton_calibration(
            source_data,
            tuple(skeleton_paths),
        )
    else:
        skeleton_ranges, skeleton_bone_lengths = None, None

    playback_identity = f"{split}:{clip_id}"
    render_clip_player(
        modalities,
        source_data,
        static_payloads,
        skeleton_ranges,
        skeleton_bone_lengths,
        playback_identity,
        row.get("duration_seconds"),
    )

    st.subheader("IMU magnitude traces")
    if imu_paths:
        figure = imu_figure(static_payloads, imu_paths)
        if figure is not None:
            st.plotly_chart(figure, width="stretch")
        else:
            st.info("IMU files exist but contain no usable samples.")
    else:
        st.info("IMU unavailable.")

    with st.expander("Clip metadata"):
        metadata = row.dropna().to_dict()
        st.json(metadata)


def _offer_label_table_reset(output_path: Path, flash_key: str) -> None:
    """Offer a way out when the manual-label file cannot be read.

    Every path through the page needs that file to parse before anything
    renders, so without this the only recovery from a corrupt or reordered
    file is deleting it from a shell. Renaming keeps whatever it still holds.
    """

    if not output_path.is_file():
        return
    st.caption(
        f"`{output_path}` is the file that could not be used. Moving it aside "
        "keeps its contents and restarts labeling from a blank table."
    )
    if st.button(
        "Move the manual-label file aside",
        key=f"manual_label_reset::{output_path}",
    ):
        try:
            backup = archive_manual_label_file(output_path)
        except OSError as exc:
            st.error(f"Could not move the file aside: {exc}")
            return
        st.session_state[flash_key] = (
            "success",
            f"Moved the unreadable label file to {backup.name}.",
        )
        st.rerun()


def render_manual_labeling(
    dataset_root: Path,
    test_csv: Path,
    sources: dict[str, dict[str, str]],
) -> None:
    """Render a resumable, one-clip-at-a-time test labeling page."""

    st.header("Manual test-data labeling")
    st.caption(
        "Review each multimodal clip, assign one of the 40 actions, and save "
        "without changing the original test index."
    )

    source_data = sources.get("test")
    if source_data is None:
        st.error("No test data source was found under the active dataset root.")
        return
    source = source_from_dict(source_data)
    if not source.readable:
        st.error("The test data source is not readable, so clips cannot be labeled.")
        return
    if not test_csv.is_file():
        st.error(f"Test CSV does not exist: {test_csv}")
        return
    if not CANONICAL_TEST_CSV.is_file():
        st.error(f"The tracked test index is missing: {CANONICAL_TEST_CSV}")
        return

    mapping_candidates = (
        dataset_root / "Training" / "class_mapping.csv",
        REPOSITORY_ROOT / "Training" / "class_mapping.csv",
    )
    mapping_path = next((path for path in mapping_candidates if path.is_file()), None)
    if mapping_path is None:
        st.error("Training/class_mapping.csv was not found in the dataset or repository.")
        return

    output_path = manual_label_path(CANONICAL_TEST_CSV)
    scope = str(output_path.resolve())
    selector_key = f"manual_label_clip::{scope}"
    pending_key = f"manual_label_pending_clip::{scope}"
    flash_key = f"manual_label_flash::{scope}"

    try:
        action_mapping = load_action_mapping(mapping_path)
        fieldnames, rows = load_manual_label_rows(
            CANONICAL_TEST_CSV,
            output_path,
            valid_action_ids=action_mapping,
        )
        # The table always spans the tracked index; the active dataset decides
        # which of its rows are reachable from this page.
        indices = active_row_indices(rows, test_csv)
        visible_rows = [rows[index] for index in indices]
        clip_ids = [clip_id_from_submission_path(row["path"]) for row in visible_rows]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("The test CSV contains duplicate clip IDs")
    except Exception as exc:
        st.error(f"Could not load the labeling table: {exc}")
        _offer_label_table_reset(output_path, flash_key)
        return
    if not visible_rows:
        st.info("The active test CSV contains no clips.")
        return

    labeled_count = sum(bool(row["prediction"].strip()) for row in visible_rows)
    st.progress(labeled_count / len(visible_rows))
    st.caption(
        f"{labeled_count:,} of {len(visible_rows):,} clips labeled · "
        f"source `{test_csv}` · output `{output_path}`"
    )
    if len(visible_rows) != len(rows):
        recorded = sum(bool(row["prediction"].strip()) for row in rows)
        st.caption(
            f"This dataset reaches {len(visible_rows):,} of the {len(rows):,} "
            f"clips in the tracked test index. Every dataset writes the same "
            f"file, which holds {recorded:,} labels in total."
        )

    pending_clip = st.session_state.pop(pending_key, None)
    if pending_clip in clip_ids:
        st.session_state[selector_key] = pending_clip
    if st.session_state.get(selector_key) not in clip_ids:
        st.session_state[selector_key] = initial_label_clip(clip_ids, visible_rows)

    flash = st.session_state.pop(flash_key, None)
    if flash:
        level, message = flash
        getattr(st, level)(message)

    row_by_clip = dict(zip(clip_ids, visible_rows))

    def clip_label(clip_id: str) -> str:
        prediction = row_by_clip[clip_id]["prediction"].strip()
        if not prediction:
            return f"{clip_id} · unlabeled"
        action_id = int(prediction)
        return f"{clip_id} · {prediction_label(action_id, action_mapping[action_id])}"

    clip_id = st.selectbox(
        "Test clip",
        clip_ids,
        format_func=clip_label,
        key=selector_key,
    )
    row_index = clip_ids.index(clip_id)
    # ``visible_rows`` holds the same dicts as ``rows``, so writing through
    # either index updates the canonical table that gets saved.
    table_index = indices[row_index]
    current_prediction = visible_rows[row_index]["prediction"].strip()
    current_action = int(current_prediction) if current_prediction else None
    action_options: list[int | None] = [None, *sorted(action_mapping)]
    # The saved prediction belongs in the key, not just in ``index``: a value
    # already in session_state under a widget key overrides ``index``, so
    # reusing one key per clip would keep showing the pre-save selection after
    # a save or a clear. Because the key and the index derive from the same
    # value, an unchanged key always implies an unchanged index.
    action_key = (
        f"manual_label_action::{scope}::{clip_id}::"
        f"{current_prediction or 'blank'}"
    )
    selected_action = st.selectbox(
        "Action label",
        action_options,
        index=action_options.index(current_action),
        format_func=lambda value: (
            "Choose an action…"
            if value is None
            else prediction_label(value, action_mapping[value])
        ),
        key=action_key,
    )

    def queue_clip(target_clip: str) -> None:
        st.session_state[pending_key] = target_clip

    def save_label(advance: bool) -> None:
        if selected_action is None:
            return
        rows[table_index]["prediction"] = str(selected_action)
        try:
            write_manual_label_rows(
                output_path, fieldnames, rows, valid_action_ids=action_mapping
            )
        except Exception as exc:
            st.session_state[flash_key] = ("error", f"Could not save labels: {exc}")
            return
        st.session_state[flash_key] = (
            "success",
            f"Saved {clip_id} to {output_path.name}.",
        )
        if advance:
            queue_clip(next_unlabeled_clip(clip_ids, visible_rows, row_index))

    def clear_label() -> None:
        rows[table_index]["prediction"] = ""
        try:
            write_manual_label_rows(
                output_path, fieldnames, rows, valid_action_ids=action_mapping
            )
        except Exception as exc:
            st.session_state[flash_key] = ("error", f"Could not save labels: {exc}")
            return
        st.session_state[flash_key] = (
            "success",
            f"Cleared the label for {clip_id}.",
        )

    previous_clip = clip_ids[(row_index - 1) % len(clip_ids)]
    next_clip = clip_ids[(row_index + 1) % len(clip_ids)]
    controls = st.columns((1, 1, 1.4, 1, 1))
    controls[0].button(
        "← Previous",
        on_click=queue_clip,
        args=(previous_clip,),
        width="stretch",
    )
    controls[1].button(
        "Clear",
        on_click=clear_label,
        disabled=current_action is None,
        width="stretch",
    )
    controls[2].button(
        "Save & next unlabeled",
        on_click=save_label,
        args=(True,),
        disabled=selected_action is None,
        type="primary",
        width="stretch",
    )
    controls[3].button(
        "Save",
        on_click=save_label,
        args=(False,),
        disabled=selected_action is None,
        width="stretch",
    )
    controls[4].button(
        "Next →",
        on_click=queue_clip,
        args=(next_clip,),
        width="stretch",
    )

    if output_path.is_file():
        st.download_button(
            "Download manual labels",
            data=output_path.read_bytes(),
            file_name=output_path.name,
            mime="text/csv",
        )

    with st.spinner("Indexing clip…"):
        modalities = cached_clip_index(source_data, clip_id)
    if not modalities:
        st.warning(f"No sensor files were found for {clip_id}.")
        return

    counts = st.columns(6)
    for column, modality in zip(counts, MODALITIES):
        column.metric(modality, len(modalities.get(modality, ())))

    imu_paths = modalities.get("IMU", [])
    radar_paths = modalities.get("Radar", [])
    skeleton_paths = modalities.get("Skeleton", [])
    static_paths = tuple(dict.fromkeys([*imu_paths, *radar_paths]))
    static_payloads = cached_payloads(source_data, static_paths) if static_paths else {}
    if skeleton_paths:
        skeleton_ranges, skeleton_bone_lengths = cached_skeleton_calibration(
            source_data,
            tuple(skeleton_paths),
        )
    else:
        skeleton_ranges, skeleton_bone_lengths = None, None

    render_clip_player(
        modalities,
        source_data,
        static_payloads,
        skeleton_ranges,
        skeleton_bone_lengths,
        f"manual-label:test:{clip_id}",
        None,
    )

    st.subheader("IMU magnitude traces")
    if imu_paths:
        figure = imu_figure(static_payloads, imu_paths)
        if figure is not None:
            st.plotly_chart(figure, width="stretch")
        else:
            st.info("IMU files exist but contain no usable samples.")
    else:
        st.info("IMU unavailable.")


def render_quality(frame: pd.DataFrame) -> None:
    st.header("Data quality")
    split = st.selectbox("Quality split", sorted(frame["split"].unique()), key="quality_split")
    data = frame[frame["split"] == split].copy()

    missing = int((data["modality_count"].fillna(0) < len(MODALITIES)).sum())
    # `radar_empty` and `imu_empty_files` only exist after a deep scan reads the
    # payloads; `depth_skeleton_aligned` only exists once some clip carried both
    # timestamps. In either case an absent column means the check never ran, so
    # show "—" rather than a zero that reads as "measured, and clean".
    radar_values = data.get("radar_empty", pd.Series(False, index=data.index))
    radar_empty = int(radar_values.map(lambda value: str(value).strip().lower() == "true").sum())
    imu_empty = int(pd.to_numeric(data.get("imu_empty_files", pd.Series(0, index=data.index)), errors="coerce").fillna(0).sum())
    alignment = int((data.get("depth_skeleton_aligned", pd.Series(True, index=data.index)) == False).sum())  # noqa: E712

    checked = {"radar_empty", "imu_empty_files", "depth_skeleton_aligned"} & set(data.columns)

    def measured(column: str, value: int) -> str:
        return f"{value:,}" if column in checked else "—"

    columns = st.columns(4)
    columns[0].metric("Incomplete clips", f"{missing:,}")
    columns[1].metric("Empty radar clips", measured("radar_empty", radar_empty))
    columns[2].metric("Empty IMU files", measured("imu_empty_files", imu_empty))
    columns[3].metric("Depth/Skeleton mismatches", measured("depth_skeleton_aligned", alignment))
    if len(checked) < 3:
        st.caption("“—” marks a check that was not run at the current scan depth.")

    left, right = st.columns(2)
    with left:
        st.subheader("Missing modality counts")
        rows = []
        for modality in MODALITIES:
            column = f"{MODALITY_SLUGS[modality]}_present"
            rows.append({"modality": modality, "missing": int((~data[column].fillna(False).astype(bool)).sum())})
        figure = px.bar(pd.DataFrame(rows), x="modality", y="missing", text="missing")
        figure.update_layout(height=400)
        st.plotly_chart(figure, width="stretch")
    with right:
        st.subheader("Issue categories")
        issue_counts: Counter[str] = Counter()
        for issues in data["issues"].fillna(""):
            for issue in str(issues).split("; "):
                if issue:
                    issue_counts[issue] += 1
        issue_frame = pd.DataFrame(issue_counts.most_common(), columns=["issue", "clips"])
        if issue_frame.empty:
            st.success("No issues were recorded at the selected scan depth.")
        else:
            figure = px.bar(issue_frame, x="clips", y="issue", orientation="h")
            figure.update_layout(height=max(400, 28 * len(issue_frame)), yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(figure, width="stretch")

    duration = pd.to_numeric(data.get("duration_seconds"), errors="coerce").dropna()
    if not duration.empty:
        st.subheader("Clip duration distribution")
        figure = px.histogram(duration, nbins=50, labels={"value": "seconds"})
        figure.update_layout(showlegend=False, height=360)
        st.plotly_chart(figure, width="stretch")

    issue_rows = data[data["issues"].fillna("") != ""].copy()
    st.subheader("Flagged clips")
    if issue_rows.empty:
        st.info("No clips are currently flagged.")
    else:
        display_columns = [
            column
            for column in ("clip_id", "action_name", "user", "trial", "missing_modalities", "issues")
            if column in issue_rows
        ]
        st.dataframe(issue_rows[display_columns], hide_index=True, width="stretch")


def render_background_manifest_status(
    process: subprocess.Popen[str],
    output_path: Path,
    progress_path: Path,
    visible_clips: int,
) -> None:
    """Poll a background manifest process without rerunning the active page."""

    @st.fragment(run_every=1.0)
    def status_fragment() -> None:
        return_code = process.poll()
        if return_code is None:
            progress: dict[str, object] = {}
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

            phase = str(progress.get("stage", "starting"))
            processed = int(progress.get("current", 0) or 0)
            total = int(progress.get("total", 0) or 0)
            if phase == "counting":
                st.progress(
                    0.0,
                    text=f"Showing {visible_clips:,} clips · Counting files for the full index…",
                )
            elif phase == "writing":
                st.progress(
                    0.99,
                    text=f"Showing {visible_clips:,} clips · Saving the complete index…",
                )
            elif total > 0:
                fraction = min(0.99, max(0.0, processed / total))
                split_label = phase.capitalize() if phase in {"train", "test"} else "Dataset"
                st.progress(
                    fraction,
                    text=(
                        f"Showing {visible_clips:,} clips · Indexing {split_label}: "
                        f"{processed:,}/{total:,} files"
                    ),
                )
            else:
                st.progress(
                    0.0,
                    text=f"Showing {visible_clips:,} clips · Starting the complete index…",
                )
            return

        if return_code != 0 or not output_path.is_file():
            detail = f"builder exited with status {return_code}"
            if process.stdout is not None and not process.stdout.closed:
                captured = process.stdout.read().strip()
                if captured:
                    detail = captured
            # The failed process stays cached on purpose: every widget
            # interaction triggers a full rerun, and dropping it here would
            # respawn a doomed builder each time. "Clear cached index" is the
            # explicit retry.
            st.error(f"The background dataset index failed: {detail}. Use “Clear cached index” to retry.")
            return

        # Drop the finished process from the resource cache before handing off.
        # It is keyed only by its arguments, so an exited builder would keep
        # satisfying later calls: unticking "Use saved manifest" to force a
        # rebuild would find this same completed process, immediately re-set the
        # flag below, and silently re-tick the box without rebuilding anything.
        background_manifest_process.clear()
        st.session_state["activate_generated_manifest"] = True
        st.success("Complete dataset index ready. Loading it now…")
        st.rerun(scope="app")

    status_fragment()


def main() -> None:
    st.title("CUHK-X Small Model Dataset Explorer")
    st.caption(
        "Configure the model workflow once, then inspect predictions alongside "
        "the multimodal sensor data."
    )

    repository_root = REPOSITORY_ROOT
    generated_manifest = repository_root / "artifacts" / "cuhkx_manifest.parquet"
    generated_progress = repository_root / "artifacts" / "cuhkx_manifest.progress.json"
    prediction_candidates = discover_prediction_csvs(repository_root)
    default_prediction_csv = str(prediction_candidates[0]) if prediction_candidates else ""
    default_root = str(resolve_dataset_root())
    if "prediction_csv_input" not in st.session_state:
        st.session_state["prediction_csv_input"] = default_prediction_csv
    if "workflow_prediction_output" not in st.session_state and default_prediction_csv:
        st.session_state["workflow_prediction_output"] = default_prediction_csv
    if "workflow_dataset_choice" not in st.session_state:
        st.session_state["workflow_dataset_choice"] = FULL_DATASET
    if "workflow_full_dataset_root" not in st.session_state:
        st.session_state["workflow_full_dataset_root"] = default_root

    # A fragment sets this flag when its background builder finishes. Apply
    # widget state before those widgets are instantiated on the next full run.
    if st.session_state.pop("activate_generated_manifest", False):
        st.session_state["saved_manifest_input"] = str(generated_manifest)
        st.session_state["use_manifest_checkbox"] = True
    if st.session_state.get("page_selector") == "Training pipeline":
        st.session_state["page_selector"] = "Workflow"

    # Page selector is rendered first so dataset-independent pages never trigger
    # dataset I/O or the "Loading dataset manifest…" spinner.
    with st.sidebar:
        st.header("Data source")
        page = st.radio(
            "Page",
            (
                "Workflow",
                "Overview",
                "Clip explorer",
                "Manual labeling",
                "Data quality",
                "Algorithm comparison",
            ),
            key="page_selector",
        )
        if page == "Algorithm comparison":
            st.caption("No dataset scan needed — reads `artifacts/*/validation.json`.")
            if st.button("Clear cached index"):
                st.cache_data.clear()
                st.rerun()
        elif page == "Workflow":
            st.caption("Choose data once, then extract, train, predict, and visualize.")
        elif page == "Manual labeling":
            st.caption("Assign action labels to test clips and save a resumable CSV.")
        else:
            st.caption(
                "A saved manifest loads instantly; otherwise the first 200 clips "
                "are shown while the full index builds."
            )

    if page == "Algorithm comparison":
        render_algorithm_comparison()
        return
    if page == "Workflow":
        render_training_pipeline(
            repository_root,
            default_root,
        )
        return

    # Resolve the active workflow for every dataset-dependent page.
    root, workflow_test_csv, sample_selected = workflow_dataset_paths(
        repository_root, default_root, st.session_state
    )
    dataset_root = str(root)
    st.session_state["dataset_root_input"] = dataset_root

    def open_workflow() -> None:
        st.session_state["page_selector"] = "Workflow"

    with st.sidebar:
        st.markdown("**Active workflow**")
        st.caption(
            f"Dataset: {'sample' if sample_selected else 'full'} · `{dataset_root}`"
        )
        st.button("Configure workflow", on_click=open_workflow, width="stretch")

        if page != "Manual labeling":
            default_manifest = (
                str(generated_manifest) if generated_manifest.is_file() else ""
            )
            if "saved_manifest_input" not in st.session_state:
                st.session_state["saved_manifest_input"] = default_manifest
            if "use_manifest_checkbox" not in st.session_state:
                st.session_state["use_manifest_checkbox"] = True

            # Pre-filled with the generated parquet when it exists. Clearing the
            # field or unchecking below enables progressive/background rebuilding.
            saved_manifest = st.text_input(
                "Saved manifest (optional)", key="saved_manifest_input"
            ).strip()
            use_manifest = st.checkbox(
                "Use saved manifest",
                disabled=not saved_manifest or sample_selected,
                help=(
                    "Uncheck to rebuild from the dataset. Saved full-dataset manifests "
                    "are not applied to the sample dataset."
                ),
                key="use_manifest_checkbox",
            )
            effective_manifest = (
                saved_manifest if use_manifest and not sample_selected else ""
            )
            progressive = st.checkbox(
                f"Load {INITIAL_CLIP_LIMIT} clips first",
                value=True,
                disabled=bool(effective_manifest),
                help=(
                    "Show a representative subset immediately while a complete "
                    "manifest is built in the background."
                ),
                key="progressive_loading_checkbox",
            )
            deep_test = st.checkbox(
                "Inspect test CSV/JSON quality",
                value=True,
                disabled=bool(effective_manifest),
                key="deep_test_checkbox",
            )

            if st.button("Clear cached index", key="clear_cache_main"):
                st.cache_data.clear()
                # Also drops a finished or failed background builder, so this is the
                # retry path when the complete index could not be written.
                st.cache_resource.clear()
                st.rerun()

    if page == "Manual labeling":
        if not root.is_dir():
            st.error(f"Dataset root does not exist: {root}")
            return
        try:
            label_sources = cached_sources(dataset_root)
        except Exception as exc:
            st.error(f"Could not discover the test data: {exc}")
            return
        render_manual_labeling(root, workflow_test_csv, label_sources)
        return

    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        st.error(f"Dataset root does not exist: {root}")
        st.stop()

    background_process: subprocess.Popen[str] | None = None
    partial_manifest = False
    try:
        with st.spinner("Loading dataset manifest…"):
            if effective_manifest:
                manifest = Path(effective_manifest).expanduser()
                stat = manifest.stat()
                records, sources = cached_saved_manifest(
                    str(root), effective_manifest, stat.st_mtime_ns, stat.st_size
                )
            elif progressive:
                records, sources = cached_initial_manifest(str(root), INITIAL_CLIP_LIMIT)
                if records:
                    partial_manifest = True
                    background_process = background_manifest_process(
                        str(root), str(generated_manifest), str(generated_progress), deep_test
                    )
                else:
                    records, sources = cached_live_manifest(str(root), deep_test)
            else:
                records, sources = cached_live_manifest(str(root), deep_test)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    if not records:
        st.error(
            "No clips were found. Check that Training/data or Testing/data exists "
            "under the selected root."
        )
        st.stop()

    frame = pd.DataFrame(records)

    # Predictions are now loaded *inside* Clip explorer only, so Overview/Data quality
    # stay prediction-free and do not change when switching prediction files.
    if partial_manifest and background_process is not None:
        render_background_manifest_status(
            background_process,
            generated_manifest,
            generated_progress,
            len(records),
        )
        # Only extracted directory sources can be sampled without a full scan,
        # so a split still held in an archive contributes nothing to the preview.
        # Name it explicitly — otherwise "Training clips 0" reads as a missing
        # dataset rather than a split that has not been indexed yet.
        previewed_splits = {str(record["split"]) for record in records}
        pending_splits = sorted(split for split in sources if split not in previewed_splits)
        message = (
            "This is a representative partial dataset. Overview and Data quality "
            "totals will update when the complete index is ready."
        )
        if pending_splits:
            noun = "split is" if len(pending_splits) == 1 else "splits are"
            message += (
                f" The {', '.join(pending_splits)} {noun} absent from this preview because"
                " the source is an archive; it appears when the complete index is ready."
            )
        st.warning(message)
    with st.sidebar:
        for split, source_data in sources.items():
            source = source_from_dict(source_data)
            status = "readable" if source.readable else "metadata only"
            st.caption(f"{split}: {source.kind} · {status}")

    if page == "Overview":
        render_overview(frame)
    elif page == "Clip explorer":
        render_clip_explorer(frame, sources, prediction_candidates)
    else:
        render_quality(frame)


if __name__ == "__main__":
    main()
