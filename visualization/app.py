"""Interactive Streamlit explorer for the CUHK-X small-model dataset."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping

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
from visualization.playback import (
    normalized_timeline_position,
    playback_interval,
    playback_start_frame,
    timeline_frame_count,
)
from visualization.predictions import (
    ClipPrediction,
    PredictionTable,
    discover_model_artifacts,
    discover_prediction_csvs,
    generate_all_split_predictions,
    generate_predictions_from_model,
    load_action_mapping,
    load_prediction_csv,
    parse_prediction_csv,
    prediction_csv_text,
)


st.set_page_config(page_title="CUHK-X Dataset Explorer", page_icon="🧭", layout="wide")

INITIAL_CLIP_LIMIT = 200

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


@st.cache_data(show_spinner=False)
def cached_uploaded_predictions(
    contents: bytes,
    action_mapping: tuple[tuple[int, str], ...],
) -> PredictionTable:
    return parse_prediction_csv(contents.decode("utf-8-sig"), dict(action_mapping))


def choose_member(paths: list[str], position: int) -> str | None:
    if not paths:
        return None
    index = round((len(paths) - 1) * position / 100)
    return paths[index]


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


def add_split_predictions(
    frame: pd.DataFrame,
    predictions_by_split: Mapping[str, PredictionTable],
) -> pd.DataFrame:
    """Attach predictions from a ``{split: PredictionTable}`` mapping."""

    enriched = frame.copy()
    # Build unified maps, then delegate to ``add_predictions`` so the
    # correctness logic stays in one place.
    merged: dict[str, ClipPrediction] = {}
    for table in predictions_by_split.values():
        merged.update(table.by_clip)
    combined = PredictionTable(
        by_clip=merged,
        rows_read=sum(t.rows_read for t in predictions_by_split.values()),
        blank_predictions=sum(t.blank_predictions for t in predictions_by_split.values()),
    )
    return add_predictions(enriched, combined)


def merge_prediction_sources(
    generated: Mapping[str, PredictionTable],
    csv_table: PredictionTable | None,
) -> dict[str, PredictionTable]:
    """Combine model-generated and CSV predictions into one attachable mapping.

    Model-generated predictions win on overlapping clip IDs. The result must be
    attached in a single ``add_split_predictions`` call: ``add_predictions``
    reassigns the prediction columns wholesale, so attaching the two sources in
    sequence would discard whichever was attached first.
    """

    merged = dict(generated)
    if csv_table is None:
        return merged
    generated_ids = {clip_id for table in generated.values() for clip_id in table.by_clip}
    csv_only = {
        clip_id: prediction
        for clip_id, prediction in csv_table.by_clip.items()
        if clip_id not in generated_ids
    }
    if csv_only:
        merged["csv"] = PredictionTable(
            by_clip=csv_only, rows_read=len(csv_only), blank_predictions=0
        )
    return merged


def _render_prediction_section(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Render prediction visualizations for both training and test splits."""

    has_train_preds = "prediction_action_id" in train.columns and not train.dropna(
        subset=["prediction_action_id"]
    ).empty
    has_test_preds = "prediction_action_id" in test.columns and not test.dropna(subset=["prediction_action_id"]).empty

    if not has_train_preds and not has_test_preds:
        return

    st.subheader("Model predictions")

    if has_train_preds and has_test_preds:
        train_pred = train.dropna(subset=["prediction_action_id"]).copy()
        test_pred = test.dropna(subset=["prediction_action_id"]).copy()
        cols = st.columns(4)
        cols[0].metric("Train predicted", f"{len(train_pred):,} / {len(train):,}")
        accuracy = float((train_pred["prediction_action_id"].astype(int) == train_pred["action_id"].astype(int)).mean()) if len(
            train_pred
        ) else 0.0
        cols[1].metric("Train accuracy", f"{accuracy:.1%}")
        cols[2].metric("Test predicted", f"{len(test_pred):,} / {len(test):,}")
        cols[3].metric("Predicted actions (test)", int(test_pred["prediction_action_id"].nunique()))
    elif has_test_preds:
        predicted = test.dropna(subset=["prediction_action_id"]).copy()
        prediction_metrics = st.columns(3)
        prediction_metrics[0].metric("Predicted clips", f"{len(predicted):,}")
        prediction_metrics[1].metric(
            "Visible coverage",
            f"{len(predicted) / len(test):.1%}" if len(test) else "—",
        )
        prediction_metrics[2].metric(
            "Predicted actions",
            int(predicted["prediction_action_id"].nunique()),
        )
    elif has_train_preds:
        train_pred = train.dropna(subset=["prediction_action_id"]).copy()
        cols = st.columns(3)
        cols[0].metric("Predicted training clips", f"{len(train_pred):,} / {len(train):,}")
        accuracy = float((train_pred["prediction_action_id"].astype(int) == train_pred["action_id"].astype(int)).mean()) if len(
            train_pred
        ) else 0.0
        cols[1].metric("Training accuracy", f"{accuracy:.1%}")
        cols[2].metric("Predicted actions", int(train_pred["prediction_action_id"].nunique()))

    # Training-specific visualizations
    if has_train_preds:
        train_pred = train.dropna(subset=["prediction_action_id"]).copy()
        # Accuracy and correctness breakdown
        correct = (train_pred["prediction_action_id"].astype(int) == train_pred["action_id"].astype(int)).sum()
        incorrect = len(train_pred) - int(correct)
        st.markdown("**Training predictions — correctness**")
        c1, c2 = st.columns(2)
        with c1:
            acc = float(correct / len(train_pred)) if len(train_pred) else 0.0
            st.metric("Correct", f"{int(correct):,} ({acc:.1%})")
        with c2:
            st.metric("Incorrect", f"{int(incorrect):,}")

        # Confusion-matrix style heatmap: true vs predicted
        try:
            true_ids = train_pred["action_id"].astype(int)
            pred_ids = train_pred["prediction_action_id"].astype(int)
            all_ids = sorted(set(true_ids) | set(pred_ids))
            # Build confusion matrix manually to avoid sklearn dependency if missing.
            import numpy as np

            id_to_pos = {aid: i for i, aid in enumerate(all_ids)}
            matrix = np.zeros((len(all_ids), len(all_ids)), dtype=int)
            for t, p in zip(true_ids, pred_ids, strict=True):
                matrix[id_to_pos[int(t)], id_to_pos[int(p)]] += 1
            # Labels for display; one pass over the frame instead of a filter per class.
            name_by_id = dict(zip(true_ids, train_pred["action_name"], strict=True))
            labels = [pretty_action(name_by_id[aid]) if aid in name_by_id else str(aid) for aid in all_ids]
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    colorbar_title="count",
                    hovertemplate="true=%{y}<br>pred=%{x}<br>count=%{z}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Training confusion matrix (true vs predicted)",
                xaxis_title="Predicted",
                yaxis_title="True",
                height=520,
                xaxis={"tickangle": -45},
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("Rows = true action, columns = predicted. Only training clips with a prediction are shown.")
        except Exception as exc:
            st.warning(f"Could not render the training confusion matrix: {exc}")

        # Distribution of predicted vs true for training
        pred_dist = (
            train_pred.groupby(["prediction_action_id", "prediction_action_name"], dropna=False)
            .size()
            .reset_index(name="clips")
            .sort_values("prediction_action_id")
        )
        pred_dist["action"] = pred_dist.apply(
            lambda row: prediction_label(row["prediction_action_id"], row["prediction_action_name"]),
            axis=1,
        )
        true_dist = (
            train_pred.groupby(["action_id", "action_name"], dropna=False)
            .size()
            .reset_index(name="clips")
            .sort_values("action_id")
        )
        true_dist["action"] = true_dist["action_name"].map(pretty_action)
        # Side-by-side bar comparison
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(pred_dist, x="action", y="clips", color="clips", color_continuous_scale="Teal", title="Predicted distribution (train)")
            fig.update_layout(xaxis_tickangle=-55, coloraxis_showscale=False, height=430)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.bar(true_dist, x="action", y="clips", color="clips", color_continuous_scale="Blues", title="True distribution (train)")
            fig.update_layout(xaxis_tickangle=-55, coloraxis_showscale=False, height=430)
            st.plotly_chart(fig, width="stretch")

        with st.expander("Predictions by training clip"):
            prediction_rows = train[["clip_id", "action_id", "action_name", "prediction_action_id", "prediction_action_name"]].copy()
            prediction_rows = prediction_rows.sort_values("clip_id")
            prediction_rows["true_action"] = prediction_rows.apply(
                lambda row: prediction_label(row["action_id"], row["action_name"]) if pd.notna(row["action_id"]) else "Unknown",
                axis=1,
            )
            prediction_rows["predicted_action"] = prediction_rows.apply(
                lambda row: prediction_label(row["prediction_action_id"], row["prediction_action_name"]) if pd.notna(row["prediction_action_id"]) else "No prediction",
                axis=1,
            )
            prediction_rows["correct"] = prediction_rows.apply(
                lambda row: "✓" if pd.notna(row["prediction_action_id"]) and row["prediction_action_id"] == row["action_id"] else ("✗" if pd.notna(row["prediction_action_id"]) else "—"),
                axis=1,
            )
            st.dataframe(
                prediction_rows[["clip_id", "true_action", "predicted_action", "correct"]],
                hide_index=True,
                width="stretch",
            )

    if has_test_preds:
        predicted = test.dropna(subset=["prediction_action_id"]).copy()
        distribution = (
            predicted.groupby(["prediction_action_id", "prediction_action_name"], dropna=False)
            .size()
            .reset_index(name="clips")
            .sort_values("prediction_action_id")
        )
        distribution["action"] = distribution.apply(
            lambda row: prediction_label(row["prediction_action_id"], row["prediction_action_name"]),
            axis=1,
        )
        st.markdown("**Test predictions — distribution**")
        figure = px.bar(
            distribution,
            x="action",
            y="clips",
            color="clips",
            color_continuous_scale="Teal",
        )
        figure.update_layout(xaxis_tickangle=-55, coloraxis_showscale=False, height=430)
        st.plotly_chart(figure, width="stretch")

        with st.expander("Predictions by test clip"):
            prediction_rows = test[["clip_id", "prediction_action_id", "prediction_action_name"]].sort_values("clip_id")
            prediction_rows = prediction_rows.rename(
                columns={
                    "prediction_action_id": "action_id",
                    "prediction_action_name": "action_name",
                }
            )
            prediction_rows["action"] = prediction_rows.apply(
                lambda row: prediction_label(row["action_id"], row["action_name"]) if pd.notna(row["action_id"]) else "No prediction",
                axis=1,
            )
            st.dataframe(
                prediction_rows[["clip_id", "action_id", "action"]],
                hide_index=True,
                width="stretch",
            )


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

    _render_prediction_section(train, test)

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


def skeleton_axis_ranges(
    payloads: Iterable[bytes],
) -> SkeletonAxisRanges | None:
    """Return one equal-scale axis cube covering every pose in a clip."""

    points = [
        point
        for payload in payloads
        for person_points, _ in _skeleton_people(payload)
        for point in person_points
    ]
    return _skeleton_axis_ranges_from_points(points)


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


def skeleton_figure(
    data: bytes,
    axis_ranges: SkeletonAxisRanges | None = None,
    uirevision: str = "skeleton",
) -> go.Figure | None:
    people = _skeleton_people(data)
    if not people:
        return None
    figure = go.Figure()
    plotted_points: list[tuple[float, float, float]] = []
    for person_index, (points, scores) in enumerate(people):
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        z = [point[2] for point in points]
        plotted_points.extend(points)
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_z: list[float | None] = []
        for start, end in SKELETON_EDGES:
            edge_x.extend((x[start], x[end], None))
            edge_y.extend((y[start], y[end], None))
            edge_z.extend((z[start], z[end], None))
        figure.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line={"color": SKELETON_COLORS[person_index % len(SKELETON_COLORS)], "width": 7},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker={
                    "color": SKELETON_COLORS[person_index % len(SKELETON_COLORS)],
                    "size": [max(4, min(9, score * 8)) for score in scores],
                },
                name=f"Person {person_index + 1}",
                text=[
                    f"{name}<br>score={score:.3f}"
                    for name, score in zip(SKELETON_JOINT_NAMES, scores, strict=True)
                ],
                hoverinfo="text",
            )
        )
    if not figure.data:
        return None

    # Fall back to frame-local bounds for standalone callers. The clip explorer
    # supplies bounds computed from every frame so playback never zooms.
    axis_ranges = axis_ranges or _skeleton_axis_ranges_from_points(plotted_points)
    assert axis_ranges is not None
    figure.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        uirevision=uirevision,
        scene={
            "aspectmode": "cube",
            "uirevision": uirevision,
            "camera": {
                "up": {"x": 0, "y": 0, "z": 1},
                # Look straight along the sensor depth axis. From +y, negative
                # skeleton x appears on camera-right, matching the paired
                # Depth Color and IR frames.
                "eye": {"x": 0, "y": 2.5, "z": 0},
                "projection": {"type": "orthographic"},
            },
            "xaxis": {"title": "Horizontal (x)", "range": axis_ranges[0]},
            "yaxis": {"title": "Depth (y)", "range": axis_ranges[1]},
            "zaxis": {"title": "Height (z)", "range": axis_ranges[2]},
        },
    )
    return figure


def _skeleton_viewer_html(
    data: bytes,
    axis_ranges: SkeletonAxisRanges | None,
    bone_lengths: SkeletonBoneLengths | None,
    playback_identity: str,
) -> str | None:
    """Build an interactive skeleton viewer whose camera survives reruns."""

    people = _skeleton_people(data)
    if not people:
        return None
    if bone_lengths is not None:
        people = [
            (_stabilize_skeleton_points(person_points, bone_lengths), scores)
            for person_points, scores in people
        ]
    points = [point for person_points, _ in people for point in person_points]
    axis_ranges = axis_ranges or _skeleton_axis_ranges_from_points(points)
    if axis_ranges is None:
        return None

    model = {
        "people": [
            {
                "points": person_points,
                "scores": scores,
                "color": SKELETON_COLORS[index % len(SKELETON_COLORS)],
                "name": f"Person {index + 1}",
            }
            for index, (person_points, scores) in enumerate(people)
        ],
        "edges": SKELETON_EDGES,
        "jointNames": SKELETON_JOINT_NAMES,
        "ranges": axis_ranges,
    }
    model_json = json.dumps(model, separators=(",", ":")).replace("</", "<\\/")
    storage_key = json.dumps(f"cuhkx:skeleton-camera:{playback_identity}").replace("</", "<\\/")
    template = """
<style>
  html, body { margin: 0; overflow: hidden; background: transparent; font-family: sans-serif; }
  #viewer { position: relative; width: 100%; height: 520px; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
  canvas.dragging { cursor: grabbing; }
  #hint { position: absolute; left: 10px; top: 8px; color: #6b7280; font-size: 12px; }
  #reset { position: absolute; right: 10px; top: 8px; border: 1px solid #aeb4bd;
    border-radius: 6px; padding: 5px 9px; background: rgba(255,255,255,.86); cursor: pointer; }
  #tooltip { display: none; position: absolute; pointer-events: none; padding: 5px 7px;
    border-radius: 5px; background: rgba(30,34,40,.9); color: white; font-size: 12px; }
</style>
<div id="viewer">
  <canvas id="skeleton"></canvas>
  <div id="hint">Drag to rotate · Scroll to zoom</div>
  <button id="reset" type="button">Reset view</button>
  <div id="tooltip"></div>
</div>
<script>
(() => {
  const model = __MODEL__;
  const storageKey = __STORAGE_KEY__;
  const canvas = document.getElementById("skeleton");
  const context = canvas.getContext("2d");
  const tooltip = document.getElementById("tooltip");
  const defaultCamera = {yaw: 0, pitch: 0, zoom: 1};
  // Pitch must stay clear of straight up/down: the basis takes cross(forward,
  // [0,0,1]), which degenerates to a zero vector there and collapses every
  // projected point onto the centre. Clamp on restore too, not just on drag,
  // so stored state from another build cannot produce a blank viewer.
  const clampCamera = value => {
    const number = (input, fallback) => (Number.isFinite(input) ? input : fallback);
    return {
      yaw: number(value.yaw, 0),
      pitch: Math.max(-1.45, Math.min(1.45, number(value.pitch, 0))),
      zoom: Math.max(0.55, Math.min(2.5, number(value.zoom, 1))),
    };
  };
  let camera = {...defaultCamera};
  try {
    const stored = JSON.parse(window.parent.sessionStorage.getItem(storageKey) || "{}");
    if (stored && typeof stored === "object") camera = clampCamera({...camera, ...stored});
  } catch (_) {}
  let dragging = false;
  let lastPointer = null;
  let projectedJoints = [];

  const add = (a, b) => a.map((value, index) => value + b[index]);
  const scale = (value, amount) => value.map(item => item * amount);
  const dot = (a, b) => a.reduce((total, value, index) => total + value * b[index], 0);
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const normalize = value => {
    const length = Math.sqrt(dot(value, value)) || 1;
    return scale(value, 1 / length);
  };
  const centers = model.ranges.map(range => (range[0] + range[1]) / 2);
  const halfSpan = (model.ranges[0][1] - model.ranges[0][0]) / 2;

  function saveCamera() {
    try { window.parent.sessionStorage.setItem(storageKey, JSON.stringify(camera)); } catch (_) {}
  }

  function basis() {
    const horizontal = Math.cos(camera.pitch);
    const eye = normalize([
      Math.sin(camera.yaw) * horizontal,
      Math.cos(camera.yaw) * horizontal,
      Math.sin(camera.pitch),
    ]);
    const forward = scale(eye, -1);
    const right = normalize(cross(forward, [0, 0, 1]));
    return {eye, right, up: normalize(cross(right, forward))};
  }

  function project(point, width, height, cameraBasis) {
    const centered = point.map((value, index) => (value - centers[index]) / halfSpan);
    const pixels = Math.min(width, height) * 0.37 * camera.zoom;
    return {
      x: width / 2 + dot(centered, cameraBasis.right) * pixels,
      y: height / 2 - dot(centered, cameraBasis.up) * pixels,
      depth: dot(centered, cameraBasis.eye),
    };
  }

  function draw() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const cameraBasis = basis();

    const cube = [];
    for (let index = 0; index < 8; index++) {
      cube.push(project([
        centers[0] + (index & 1 ? halfSpan : -halfSpan),
        centers[1] + (index & 2 ? halfSpan : -halfSpan),
        centers[2] + (index & 4 ? halfSpan : -halfSpan),
      ], width, height, cameraBasis));
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
    for (const person of model.people) {
      const projected = person.points.map(point => project(point, width, height, cameraBasis));
      const bones = model.edges.map(edge => ({edge, depth: (projected[edge[0]].depth + projected[edge[1]].depth) / 2}));
      bones.sort((a, b) => a.depth - b.depth);
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
        projectedJoints.push({...point, radius, label: `${person.name} · ${model.jointNames[index]} · score=${person.scores[index].toFixed(3)}`});
      });
    }
  }

  canvas.addEventListener("pointerdown", event => {
    dragging = true;
    lastPointer = [event.clientX, event.clientY];
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
    tooltip.style.display = "none";
  });
  canvas.addEventListener("pointermove", event => {
    if (dragging) {
      const dx = event.clientX - lastPointer[0];
      const dy = event.clientY - lastPointer[1];
      camera.yaw -= dx * 0.008;
      camera.pitch = Math.max(-1.45, Math.min(1.45, camera.pitch + dy * 0.008));
      lastPointer = [event.clientX, event.clientY];
      saveCamera();
      draw();
      return;
    }
    const bounds = canvas.getBoundingClientRect();
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
  canvas.addEventListener("pointerup", event => {
    dragging = false;
    canvas.classList.remove("dragging");
    canvas.releasePointerCapture(event.pointerId);
    saveCamera();
  });
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    camera.zoom = Math.max(0.55, Math.min(2.5, camera.zoom * Math.exp(-event.deltaY * 0.001)));
    saveCamera();
    draw();
  }, {passive: false});
  document.getElementById("reset").addEventListener("click", () => {
    camera = {...defaultCamera};
    saveCamera();
    draw();
  });
  new ResizeObserver(draw).observe(document.getElementById("viewer"));
  draw();
})();
</script>
"""
    return template.replace("__MODEL__", model_json).replace("__STORAGE_KEY__", storage_key)


def _render_skeleton_viewer(html: str) -> None:
    if hasattr(st, "iframe"):
        st.iframe(html, width="stretch", height=520, tab_index=0)
    else:  # pragma: no cover - compatibility with older supported Streamlit
        # requirements.txt still allows streamlit<1.62, which has no st.iframe.
        # components.html sandboxes the frame, so window.parent.sessionStorage
        # throws and the camera simply resets between reruns instead of
        # persisting. The viewer itself works either way.
        components.html(html, height=520)


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


def radar_figure(data: bytes, position: int) -> tuple[go.Figure | None, int, int]:
    text = data.decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    rows = [row for row in rows if row.get("frame") not in (None, "")]
    if not rows:
        return None, 0, 0
    frames = sorted({int(float(row["frame"])) for row in rows})
    selected = frames[round((len(frames) - 1) * position / 100)]
    points = [row for row in rows if int(float(row["frame"])) == selected]

    def values(name: str) -> list[float]:
        result = []
        for row in points:
            try:
                result.append(float(row[name]))
            except (KeyError, TypeError, ValueError):
                result.append(0.0)
        return result

    snr = values("snr")
    velocity = values("v")
    figure = go.Figure(
        go.Scatter3d(
            x=values("x"),
            y=values("y"),
            z=values("z"),
            mode="markers",
            marker={
                "size": [max(3, min(11, value / 30)) for value in snr],
                "color": velocity,
                "colorscale": "RdBu",
                "cmid": 0,
                "colorbar": {"title": "velocity"},
                "opacity": 0.82,
            },
            text=[f"frame={selected}<br>SNR={snr[index]:.0f}<br>v={velocity[index]:.3f}" for index in range(len(points))],
            hoverinfo="text",
        )
    )
    figure.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        scene={"aspectmode": "data", "xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"},
    )
    return figure, selected, len(frames)


def render_visual_frame(
    modalities: dict[str, list[str]],
    source_data: dict[str, str],
    static_payloads: dict[str, bytes],
    skeleton_ranges: SkeletonAxisRanges | None,
    skeleton_bone_lengths: SkeletonBoneLengths | None,
    playback_identity: str,
    position: int,
) -> None:
    """Render all modalities that change as the playback cursor advances."""

    selected_images = {
        modality: choose_member(modalities.get(modality, []), position)
        for modality in ("Depth_Color", "IR", "Thermal")
    }
    skeleton_path = choose_member(modalities.get("Skeleton", []), position)
    radar_paths = modalities.get("Radar", [])
    wanted = [path for path in selected_images.values() if path]
    if skeleton_path:
        wanted.append(skeleton_path)
    frame_payloads = cached_payloads(source_data, tuple(dict.fromkeys(wanted))) if wanted else {}
    payloads = {**static_payloads, **frame_payloads}

    st.subheader("Visual streams")
    image_columns = st.columns(3)
    for column, modality in zip(image_columns, ("Depth_Color", "IR", "Thermal")):
        with column:
            st.markdown(f"**{modality}**")
            path = selected_images[modality]
            if path:
                st.image(payloads[path], caption=PurePosixPath(path).name, width="stretch")
            else:
                st.info("Unavailable")

    skeleton_column, radar_column = st.columns(2)
    with skeleton_column:
        st.subheader("3D skeleton")
        if skeleton_path:
            viewer_html = _skeleton_viewer_html(
                payloads[skeleton_path],
                skeleton_ranges,
                skeleton_bone_lengths,
                playback_identity,
            )
            if viewer_html is not None:
                _render_skeleton_viewer(viewer_html)
                st.caption(
                    "Human3.6M 17-joint pose · camera-aligned · clip-stabilized limb lengths."
                )
            else:
                st.info("No valid person pose in this frame.")
        else:
            st.info("Skeleton unavailable.")
    with radar_column:
        st.subheader("Radar point cloud")
        if radar_paths:
            figure, selected_frame, total_frames = radar_figure(payloads[radar_paths[0]], position)
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
                st.caption(f"Radar frame {selected_frame}; {total_frames} frames contain detections.")
            else:
                st.info("The radar file exists but contains no detections.")
        else:
            st.info("Radar unavailable.")


def render_clip_explorer(frame: pd.DataFrame, sources: dict[str, dict[str, str]]) -> None:
    st.header("Multimodal clip explorer")
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

    frame_count = timeline_frame_count(modalities)
    playback_identity = f"{split}:{clip_id}"
    frame_key = f"playback_frame:{playback_identity}"
    if st.session_state.get("playback_identity") != playback_identity:
        st.session_state.playback_identity = playback_identity
        st.session_state.playback_running = False
        st.session_state.playback_hold_tick = False
        st.session_state[frame_key] = 0
    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = 1.0

    controls = st.columns((1, 1, 1.5, 5))
    running = bool(st.session_state.get("playback_running", False))
    current_frame = min(int(st.session_state.get(frame_key, 0)), frame_count - 1)
    # A single-frame clip is always "at the end", but it has never been played,
    # so offering Replay rather than Play would be misleading.
    finished = frame_count > 1 and current_frame >= frame_count - 1
    toggle_label = "⏸ Pause" if running else ("↻ Replay" if finished else "▶ Play")
    # Render the persistent widget before a button handler can call st.rerun().
    # Otherwise Streamlit's widget cleanup drops its state during that partial
    # pass and the next run recreates the speed at the default 1× value.
    speed = controls[2].select_slider(
        "Speed",
        options=(0.5, 1.0, 2.0),
        format_func=lambda value: f"{value:g}×",
        key="playback_speed",
    )
    if controls[0].button(toggle_label, key=f"playback_toggle:{playback_identity}", width="stretch"):
        if not running:
            st.session_state[frame_key] = playback_start_frame(current_frame, frame_count)
        st.session_state.playback_running = not running
        st.session_state.playback_hold_tick = not running
        st.rerun()
    if controls[1].button("↺ Restart", key=f"playback_restart:{playback_identity}", width="stretch"):
        st.session_state[frame_key] = 0
        st.session_state.playback_running = False
        st.session_state.playback_hold_tick = False
        st.rerun()
    interval = playback_interval(row.get("duration_seconds"), frame_count, float(speed)) if running else None

    @st.fragment(run_every=interval)
    def playback_fragment() -> None:
        current = min(int(st.session_state.get(frame_key, 0)), frame_count - 1)
        if st.session_state.get("playback_running", False):
            if st.session_state.get("playback_hold_tick", False):
                st.session_state.playback_hold_tick = False
            elif current < frame_count - 1:
                current += 1
                st.session_state[frame_key] = current
            else:
                st.session_state.playback_running = False
                st.rerun()

        def hold_after_seek() -> None:
            st.session_state.playback_hold_tick = True

        current = st.slider(
            "Timeline frame",
            min_value=0,
            max_value=frame_count - 1,
            key=frame_key,
            on_change=hold_after_seek,
            help="Drag to scrub manually; Play advances this timeline automatically.",
        )
        position = normalized_timeline_position(current, frame_count)
        st.caption(f"Frame {current + 1} of {frame_count} · {position}% through clip · {float(speed):g}×")
        render_visual_frame(
            modalities,
            source_data,
            static_payloads,
            skeleton_ranges,
            skeleton_bone_lengths,
            playback_identity,
            position,
        )

    playback_fragment()

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

            phase = str(progress.get("phase", "starting"))
            processed = int(progress.get("processed", 0) or 0)
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
        "Overview, synchronized multimodal samples, predicted test actions, "
        "and data-quality diagnostics."
    )

    repository_root = REPOSITORY_ROOT
    generated_manifest = repository_root / "artifacts" / "cuhkx_manifest.parquet"
    generated_progress = repository_root / "artifacts" / "cuhkx_manifest.progress.json"
    prediction_candidates = discover_prediction_csvs(repository_root)
    default_prediction_csv = str(prediction_candidates[0]) if prediction_candidates else ""

    # A fragment sets this flag when its background builder finishes. Apply
    # widget state before those widgets are instantiated on the next full run.
    if st.session_state.pop("activate_generated_manifest", False):
        st.session_state["saved_manifest_input"] = str(generated_manifest)
        st.session_state["use_manifest_checkbox"] = True

    # Page selector is rendered first so "Algorithm comparison" never triggers
    # dataset I/O or the "Loading dataset manifest…" spinner.
    with st.sidebar:
        st.header("Data source")
        page = st.radio("Page", ("Overview", "Clip explorer", "Data quality", "Algorithm comparison"))
        if page == "Algorithm comparison":
            st.caption("No dataset scan needed — reads `artifacts/*/validation.json`.")
            if st.button("Clear cached index"):
                st.cache_data.clear()
                st.rerun()
        else:
            st.caption(
                "A saved manifest loads instantly; otherwise the first 200 clips "
                "are shown while the full index builds."
            )

    if page == "Algorithm comparison":
        render_algorithm_comparison()
        return

    # Only for the three dataset-dependent pages
    default_root = str(resolve_dataset_root())
    default_manifest = str(generated_manifest) if generated_manifest.is_file() else ""
    if "saved_manifest_input" not in st.session_state:
        st.session_state["saved_manifest_input"] = default_manifest
    if "use_manifest_checkbox" not in st.session_state:
        st.session_state["use_manifest_checkbox"] = True
    if "prediction_csv_input" not in st.session_state:
        st.session_state["prediction_csv_input"] = default_prediction_csv
    with st.sidebar:
        dataset_root = st.text_input("Dataset root", default_root, key="dataset_root_input")
        # Pre-filled with the generated parquet when it exists. Clearing the
        # field or unchecking below enables progressive/background rebuilding.
        saved_manifest = st.text_input(
            "Saved manifest (optional)", key="saved_manifest_input"
        ).strip()
        use_manifest = st.checkbox(
            "Use saved manifest",
            disabled=not saved_manifest,
            help="Uncheck to rebuild from the dataset using progressive or blocking loading below.",
            key="use_manifest_checkbox",
        )
        effective_manifest = saved_manifest if use_manifest else ""
        progressive = st.checkbox(
            f"Load {INITIAL_CLIP_LIMIT} clips first",
            value=True,
            disabled=bool(effective_manifest),
            help="Show a representative subset immediately while a complete manifest is built in the background.",
            key="progressive_loading_checkbox",
        )
        deep_test = st.checkbox(
            "Inspect test CSV/JSON quality",
            value=True,
            disabled=bool(effective_manifest),
            key="deep_test_checkbox",
        )
        st.subheader("Test predictions")
        prediction_csv = st.text_input(
            "Predictions CSV (optional)",
            key="prediction_csv_input",
            help=(
                "A path,prediction CSV such as outputs/logreg_submission.csv. "
                "The newest *_submission.csv under outputs/ is selected automatically."
            ),
        ).strip()
        uploaded_prediction = st.file_uploader(
            "Or upload predictions CSV",
            type=("csv",),
            help="An uploaded file overrides the path above for this session.",
        )

        st.subheader("Generate predictions")
        st.caption("Run a saved model to generate predictions for training and test clips.")
        model_artifacts = discover_model_artifacts(repository_root)
        if model_artifacts:
            artifact_options = [str(path) for path in model_artifacts]
            # Default to the newest artifact.
            default_artifact = artifact_options[0] if artifact_options else ""
            if "model_artifact_input" not in st.session_state:
                st.session_state["model_artifact_input"] = default_artifact
            selected_model = st.selectbox(
                "Model artifact",
                artifact_options,
                key="model_artifact_input",
                help="Artifacts from `python -m modeling.train --algorithm <name>` (artifacts/*/model.joblib).",
            )
            split_choice = st.selectbox(
                "Split to predict",
                ["Both (train + test)", "Train only", "Test only"],
                key="model_split_choice",
                help="Generate predictions for the selected split(s). 'Both' enables Overview and Clip explorer visualizations for training and test.",
            )
            n_jobs_model = st.number_input(
                "Parallel jobs",
                min_value=1,
                max_value=32,
                value=4,
                step=1,
                key="model_n_jobs",
                help="Parallelism for feature extraction.",
            )
            col_gen, col_clear = st.columns(2)
            with col_gen:
                generate_clicked = st.button(
                    "Generate predictions",
                    key="generate_model_predictions",
                    help="Extract features and run the selected model. Uses the current Dataset root.",
                    width="stretch",
                )
            with col_clear:
                if st.button("Clear generated", key="clear_generated_predictions", width="stretch"):
                    for key in ("generated_model_predictions", "generated_model_source", "generated_model_split"):
                        st.session_state.pop(key, None)
                    st.cache_data.clear()
                    st.rerun()
            if generate_clicked:
                # Store request; actual generation happens after dataset manifest is loaded
                # so that dataset_root and mapping resolution are consistent.
                st.session_state["generate_requested"] = {
                    "model_path": selected_model,
                    "split_choice": split_choice,
                    "n_jobs": int(n_jobs_model),
                    "dataset_root": dataset_root,
                }
                st.rerun()
        else:
            st.caption("No saved models found at `artifacts/*/model.joblib`. Train one with `python -m modeling.train --algorithm logistic_regression`.")
            generate_clicked = False

        if st.button("Clear cached index", key="clear_cache_main"):
            st.cache_data.clear()
            # Also drops a finished or failed background builder, so this is the
            # retry path when the complete index could not be written.
            st.cache_resource.clear()
            st.rerun()

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
        st.error("No clips were found. Check that Training/data or Testing/data exists under the selected root.")
        st.stop()

    frame = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # CSV-based predictions (test and optionally train)
    # ------------------------------------------------------------------
    prediction_table: PredictionTable | None = None
    prediction_source = ""
    if prediction_csv or uploaded_prediction is not None:
        mapping_candidates = (
            repository_root / "Training" / "class_mapping.csv",
            root / "Training" / "class_mapping.csv",
        )
        mapping_path = next((path for path in mapping_candidates if path.is_file()), None)
        try:
            if mapping_path is None:
                raise FileNotFoundError(
                    "Training/class_mapping.csv was not found in the repository or dataset root"
                )
            mapping_stat = mapping_path.stat()
            action_mapping = cached_action_mapping(
                str(mapping_path), mapping_stat.st_mtime_ns, mapping_stat.st_size
            )
            mapping_items = tuple(sorted(action_mapping.items()))
            if uploaded_prediction is not None:
                prediction_table = cached_uploaded_predictions(
                    uploaded_prediction.getvalue(), mapping_items
                )
                prediction_source = uploaded_prediction.name
            else:
                prediction_path = Path(prediction_csv).expanduser()
                if not prediction_path.is_absolute():
                    prediction_path = repository_root / prediction_path
                prediction_stat = prediction_path.stat()
                prediction_table = cached_prediction_file(
                    str(prediction_path),
                    prediction_stat.st_mtime_ns,
                    prediction_stat.st_size,
                    mapping_items,
                )
                prediction_source = str(prediction_path)
            # Defer attaching until after model generation decision so we can merge.
        except Exception as exc:
            st.sidebar.error(f"Could not load predictions: {exc}")
            prediction_table = None
            prediction_source = ""

    # ------------------------------------------------------------------
    # Model-generated predictions (for training and test)
    # ------------------------------------------------------------------
    # If the user requested generation via the sidebar button, run it now.
    # This is done after the manifest is loaded so that dataset_root is
    # validated and the action mapping can be resolved consistently.
    generated_predictions: dict[str, PredictionTable] | None = None
    generated_source = ""
    if st.session_state.get("generate_requested") is not None:
        request = st.session_state.pop("generate_requested")
        model_path_req = Path(request["model_path"]).expanduser()
        split_choice_req = request.get("split_choice", "Both (train + test)")
        n_jobs_req = int(request.get("n_jobs", 4))
        # Dataset root at generation time may differ from current sidebar value;
        # use the requested root for reproducibility.
        dataset_root_req = str(request.get("dataset_root", str(root)))
        mapping_candidates_gen = (
            repository_root / "Training" / "class_mapping.csv",
            Path(dataset_root_req).expanduser() / "Training" / "class_mapping.csv",
        )
        mapping_path_gen = next((p for p in mapping_candidates_gen if p.is_file()), None)
        try:
            if mapping_path_gen is None:
                raise FileNotFoundError(
                    "Training/class_mapping.csv not found in repository or dataset root"
                )
            mapping_stat_gen = mapping_path_gen.stat()
            action_mapping_gen = cached_action_mapping(
                str(mapping_path_gen), mapping_stat_gen.st_mtime_ns, mapping_stat_gen.st_size
            )
            mapping_items_gen = tuple(sorted(action_mapping_gen.items()))

            # Progress bar replaces the previous spinner so the user sees
            # per-clip extraction progress instead of a static message.
            progress_bar = st.progress(0, text=f"Generating predictions with {model_path_req.name} — preparing…")
            # ``st.progress`` is thread-safe when called from the main thread,
            # which is where ``extract_feature_bundle`` invokes the callback as
            # it yields results from the ProcessPoolExecutor.
            def set_progress(percent: int, message: str) -> None:
                progress_bar.progress(max(0, min(100, int(percent))), text=message)

            def rescale(start: int, span: int) -> Callable[[int, int, str], None]:
                """Map one split's raw clip counts onto ``start..start+span``."""

                def forward(done: int, total: int, message: str) -> None:
                    set_progress(start + int(span * done / max(1, total)), message)

                return forward

            def percent_progress(current: int, total: int, message: str) -> None:
                # ``generate_all_split_predictions`` already reports a monotonic
                # 0-100 percentage spanning both splits, so pass it straight through.
                del total
                set_progress(current, message)

            # Generation runs uncached so the callback can drive the bar during the
            # ~2 min train extraction (2,785 clips) and ~20 s test extraction (405
            # clips). The result is persisted in ``st.session_state`` below, which is
            # what serves subsequent reruns.
            try:
                set_progress(2, "Loading model and class mapping…")
                if split_choice_req in {"Train only", "Test only"}:
                    only_split = "train" if split_choice_req == "Train only" else "test"
                    table = generate_predictions_from_model(
                        str(model_path_req),
                        dataset_root_req,
                        None,
                        dict(mapping_items_gen),
                        split=only_split,
                        n_jobs=n_jobs_req,
                        progress_callback=rescale(5, 90),
                    )
                    generated_predictions = {only_split: table}
                else:
                    generated_predictions = generate_all_split_predictions(
                        str(model_path_req),
                        dataset_root_req,
                        None,
                        dict(mapping_items_gen),
                        n_jobs=n_jobs_req,
                        progress_callback=percent_progress,
                    )
                set_progress(
                    100,
                    f"Generated {sum(len(t.by_clip) for t in generated_predictions.values()):,} "
                    "predictions — complete!",
                )
            finally:
                progress_bar.empty()
            generated_source = str(model_path_req)
            # Persist for subsequent reruns (filters, page switches) until cleared.
            st.session_state["generated_model_predictions"] = generated_predictions
            st.session_state["generated_model_source"] = generated_source
            st.session_state["generated_model_split"] = split_choice_req
            st.success(f"Generated {sum(len(t.by_clip) for t in generated_predictions.values()):,} predictions from {model_path_req.name}")
        except Exception as exc:
            st.sidebar.error(f"Could not generate predictions: {exc}")
            st.error(f"Could not generate predictions: {exc}")

    # Rehydrate previously generated predictions on normal reruns.
    if generated_predictions is None and "generated_model_predictions" in st.session_state:
        generated_predictions = st.session_state["generated_model_predictions"]
        generated_source = st.session_state.get("generated_model_source", "")

    # ------------------------------------------------------------------
    # Attach predictions to the frame (CSV and/or model-generated)
    # ------------------------------------------------------------------
    # Model-generated predictions take precedence over CSV for overlapping IDs.
    if generated_predictions is not None:
        # Attach the generated predictions and any non-conflicting CSV entries in a
        # single pass — see ``merge_prediction_sources``.
        frame = add_split_predictions(
            frame, merge_prediction_sources(generated_predictions, prediction_table)
        )
        # Report generated predictions
        total_generated = sum(len(t.by_clip) for t in generated_predictions.values())
        train_gen = len(generated_predictions.get("train", PredictionTable({}, 0, 0)).by_clip)
        test_gen = len(generated_predictions.get("test", PredictionTable({}, 0, 0)).by_clip)
        st.sidebar.caption(
            f"Generated {total_generated:,} predictions from model {generated_source} "
            f"(train: {train_gen:,}, test: {test_gen:,}). Matched {train_gen:,} training and {test_gen:,} test clips in current view."
        )
        # Keep rendering helpers aware of generated source for download etc.
        # For downstream reporting, keep prediction_table pointing to the test split if present.
        if prediction_table is None:
            prediction_table = generated_predictions.get("test") or next(iter(generated_predictions.values()))
            prediction_source = generated_source
        # Provide download for generated test predictions if present.
        if "test" in generated_predictions:
            st.sidebar.download_button(
                "Download generated test CSV",
                data=prediction_csv_text(generated_predictions["test"]),
                file_name="generated_test_predictions.csv",
                mime="text/csv",
                key="download_generated_test_csv",
            )
        if "train" in generated_predictions:
            csv_train = io.StringIO()
            writer = csv.DictWriter(csv_train, fieldnames=("clip_id", "true_action_id", "prediction", "predicted_action_name"))
            writer.writeheader()
            train_frame = frame[frame["split"] == "train"]
            for clip_id, pred in generated_predictions["train"].by_clip.items():
                true_row = train_frame[train_frame["clip_id"] == clip_id]
                true_id = int(true_row["action_id"].iloc[0]) if not true_row.empty and pd.notna(true_row["action_id"].iloc[0]) else ""
                writer.writerow(
                    {
                        "clip_id": clip_id,
                        "true_action_id": true_id,
                        "prediction": int(pred.action_id),
                        "predicted_action_name": pred.action_name,
                    }
                )
            st.sidebar.download_button(
                "Download generated train CSV",
                data=csv_train.getvalue(),
                file_name="generated_train_predictions.csv",
                mime="text/csv",
                key="download_generated_train_csv",
            )
    elif prediction_table is not None:
        # Only CSV predictions — attach them (now supports both train and test IDs).
        frame = add_predictions(frame, prediction_table)
        # Report coverage per split for clarity.
        train_visible = len(set(frame.loc[frame["split"] == "train", "clip_id"].astype(str)) & set(prediction_table.by_clip))
        test_visible = len(set(frame.loc[frame["split"] == "test", "clip_id"].astype(str)) & set(prediction_table.by_clip))
        if prediction_csv or uploaded_prediction is not None:
            if train_visible and test_visible:
                st.sidebar.caption(
                    f"Loaded {len(prediction_table.by_clip):,} predictions from {prediction_source}. "
                    f"Matched {train_visible:,} training and {test_visible:,} test clips in current view."
                )
            elif train_visible:
                st.sidebar.caption(
                    f"Loaded {len(prediction_table.by_clip):,} predictions from {prediction_source}. "
                    f"Matched {train_visible:,} training clips."
                )
            else:
                st.sidebar.caption(
                    f"Loaded {len(prediction_table.by_clip):,} predictions from {prediction_source}. "
                    f"Matched {test_visible:,}/{len(set(frame.loc[frame['split'] == 'test', 'clip_id'].astype(str))):,} visible test clips."
                )
            if prediction_table.blank_predictions:
                st.sidebar.warning(
                    f"Ignored {prediction_table.blank_predictions:,} row(s) with blank predictions."
                )
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
        render_clip_explorer(frame, sources)
    else:
        render_quality(frame)


if __name__ == "__main__":
    main()
