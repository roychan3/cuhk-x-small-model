"""Interactive Streamlit explorer for the CUHK-X small-model dataset."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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
from visualization.playback import normalized_timeline_position, playback_interval, timeline_frame_count


st.set_page_config(page_title="CUHK-X Dataset Explorer", page_icon="🧭", layout="wide")

INITIAL_CLIP_LIMIT = 200

SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


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


def skeleton_figure(data: bytes) -> go.Figure | None:
    try:
        people = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(people, list) or not people:
        return None
    figure = go.Figure()
    for person_index, person in enumerate(people):
        keypoints = person.get("keypoints", [])
        if len(keypoints) < 17:
            continue
        scores = person.get("keypoint_scores", [1.0] * len(keypoints))
        x = [point[0] for point in keypoints]
        y = [point[1] for point in keypoints]
        z = [point[2] for point in keypoints]
        figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker={"size": [max(4, float(score) * 8) for score in scores]},
                name=f"Person {person_index + 1}",
                text=[f"Joint {index}<br>score={scores[index]:.3f}" for index in range(len(keypoints))],
                hoverinfo="text",
            )
        )
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
                line={"width": 5},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if not figure.data:
        return None
    figure.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
        },
    )
    return figure


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
            figure = skeleton_figure(payloads[skeleton_path])
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
                st.caption("Connectivity assumes the common COCO 17-joint ordering.")
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

    clip_ids = sorted(filtered["clip_id"].dropna().unique(), key=natural_key)
    if not clip_ids:
        st.info("No clips match the filters.")
        return
    clip_id = st.selectbox("Clip", clip_ids)
    row = filtered[filtered["clip_id"] == clip_id].iloc[0]

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
    static_paths = tuple(dict.fromkeys([*imu_paths, *radar_paths]))
    static_payloads = cached_payloads(source_data, static_paths) if static_paths else {}

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
    toggle_label = "⏸ Pause" if running else "▶ Play"
    if controls[0].button(toggle_label, key=f"playback_toggle:{playback_identity}", width="stretch"):
        st.session_state.playback_running = not running
        st.session_state.playback_hold_tick = not running
        st.rerun()
    if controls[1].button("↺ Restart", key=f"playback_restart:{playback_identity}", width="stretch"):
        st.session_state[frame_key] = 0
        st.session_state.playback_running = False
        st.session_state.playback_hold_tick = False
        st.rerun()
    speed = controls[2].select_slider(
        "Speed",
        options=(0.5, 1.0, 2.0),
        format_func=lambda value: f"{value:g}×",
        key="playback_speed",
    )
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
        render_visual_frame(modalities, source_data, static_payloads, position)

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
    st.caption("Overview, synchronized multimodal samples, and data-quality diagnostics.")

    repository_root = Path(__file__).resolve().parents[1]
    generated_manifest = repository_root / "artifacts" / "cuhkx_manifest.parquet"
    generated_progress = repository_root / "artifacts" / "cuhkx_manifest.progress.json"

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
