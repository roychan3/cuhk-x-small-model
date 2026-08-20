"""Visualization dataset discovery, indexing, and manifest generation.

The training archive supplied with CUHK-X is a multi-volume ZIP. Python's
``zipfile`` module cannot read its payload directly, so this module uses
``zipinfo`` for metadata-only indexing and marks that source as unreadable.
Merged ZIPs, the test ZIP, and extracted directories are fully readable.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import uuid
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Iterator, Mapping, Sequence


MODALITIES = ("Depth_Color", "IR", "Thermal", "IMU", "Radar", "Skeleton")
MODALITY_SLUGS = {name: name.lower() for name in MODALITIES}

DATASET_ROOT_ENV_VAR = "CUHKX_DATASET_ROOT"
DEFAULT_DATASET_ROOT = Path.home() / "AAI" / "opt-scratch" / "small-model"

_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})")
_FRAME_RE = re.compile(r"(?:frame_|_)(\d+)(?:_Color)?\.(?:jpg|jpeg|png|json)$", re.I)
_ACTION_RE = re.compile(r"^(\d+)_(.+)$")


@dataclass(frozen=True)
class DataSource:
    """One physical source for a dataset split."""

    split: str
    kind: str
    path: str

    @property
    def readable(self) -> bool:
        return self.kind in {"directory", "zip"}

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Member:
    path: str
    size: int | None = None


@dataclass(frozen=True)
class ParsedMember:
    split: str
    clip_id: str
    modality: str
    member_path: str
    size: int | None
    action_id: int | None = None
    action_name: str | None = None
    user: str | None = None
    trial: str | None = None
    timestamp: str | None = None
    frame_index: int | None = None


def _supports_zipfile(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            archive.infolist()
        return True
    except (OSError, zipfile.BadZipFile):
        return False


def resolve_dataset_root(explicit: str | Path | None = None) -> Path:
    """Resolve the dataset root from an explicit value, the environment, or the default.

    Precedence is explicit argument, then ``$CUHKX_DATASET_ROOT``, then
    ``DEFAULT_DATASET_ROOT``. Every entry point resolves the root through this
    function so the command line, the dashboard, and Docker agree.
    """

    if explicit is not None and str(explicit) != "":
        return Path(explicit).expanduser()
    from_environment = os.environ.get(DATASET_ROOT_ENV_VAR)
    if from_environment:
        return Path(from_environment).expanduser()
    return DEFAULT_DATASET_ROOT


def discover_sources(dataset_root: str | Path) -> dict[str, DataSource]:
    """Discover the best available physical source for each split."""

    root = Path(dataset_root).expanduser().resolve()
    sources: dict[str, DataSource] = {}

    train_data = root / "Training" / "data"
    extracted_train = train_data / "HAR"
    if (extracted_train / "data").is_dir():
        sources["train"] = DataSource("train", "directory", str(extracted_train))
    elif train_data.is_dir():
        merged_candidates = sorted(train_data.glob("HAR_full*.zip"))
        merged_candidates += sorted(train_data.glob("HAR_merged*.zip"))
        merged = next((path for path in merged_candidates if _supports_zipfile(path)), None)
        if merged is not None:
            sources["train"] = DataSource("train", "zip", str(merged))
        elif (train_data / "HAR.zip").is_file():
            final_volume = train_data / "HAR.zip"
            kind = "zip" if _supports_zipfile(final_volume) else "multipart_zip"
            sources["train"] = DataSource("train", kind, str(final_volume))

    test_data = root / "Testing" / "data"
    extracted_test = test_data / "small_model_track_test"
    if extracted_test.is_dir():
        sources["test"] = DataSource("test", "directory", str(extracted_test))
    elif test_data.is_dir():
        candidates = sorted(test_data.glob("small_model_track_test*.zip"))
        archive = next((path for path in candidates if _supports_zipfile(path)), None)
        if archive is not None:
            sources["test"] = DataSource("test", "zip", str(archive))

    return sources


def _iter_directory_members(source: DataSource) -> Iterator[Member]:
    root = Path(source.path)
    relative_root = root.parent
    for path in root.rglob("*"):
        if path.is_file():
            yield Member(path.relative_to(relative_root).as_posix(), path.stat().st_size)


def _iter_zip_members(source: DataSource) -> Iterator[Member]:
    with zipfile.ZipFile(source.path) as archive:
        for info in archive.infolist():
            if not info.is_dir():
                yield Member(info.filename, info.file_size)


def _iter_multipart_members(source: DataSource) -> Iterator[Member]:
    executable = shutil.which("zipinfo") or shutil.which("unzip")
    if executable is None:
        raise RuntimeError(
            "The multipart training archive requires the 'zipinfo' or 'unzip' command "
            "for metadata indexing."
        )
    command = [executable, "-1", source.path] if Path(executable).name == "zipinfo" else [executable, "-Z1", source.path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None
    for raw_line in process.stdout:
        name = raw_line.rstrip(b"\r\n").decode("utf-8", errors="replace")
        if name and not name.endswith("/"):
            yield Member(name, None)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Could not index multipart archive: {source.path}")


def iter_source_members(source: DataSource) -> Iterator[Member]:
    if source.kind == "directory":
        yield from _iter_directory_members(source)
    elif source.kind == "zip":
        yield from _iter_zip_members(source)
    elif source.kind == "multipart_zip":
        yield from _iter_multipart_members(source)
    else:
        raise ValueError(f"Unsupported source kind: {source.kind}")


def _is_ignored_member(path: PurePosixPath) -> bool:
    return (
        "__MACOSX" in path.parts
        or any(part.startswith(".") for part in path.parts)
        or path.name in {"settings.local.json", "Thumbs.db"}
    )


def parse_member(source: DataSource, member: Member) -> ParsedMember | None:
    """Parse a physical file path into its logical clip and modality."""

    path = PurePosixPath(member.path)
    if _is_ignored_member(path):
        return None
    parts = path.parts

    action_id: int | None = None
    action_name: str | None = None
    user: str | None = None
    trial: str | None = None

    if source.split == "train":
        if len(parts) < 7 or parts[:2] != ("HAR", "data") or parts[2] not in MODALITIES:
            return None
        modality, action_name, user, trial = parts[2:6]
        match = _ACTION_RE.match(action_name)
        if match:
            action_id = int(match.group(1))
        clip_id = f"{action_name}/{user}/{trial}"
    elif source.split == "test":
        if (
            len(parts) < 4
            or parts[0] != "small_model_track_test"
            or not parts[1].startswith("SM_test_")
            or parts[2] not in MODALITIES
        ):
            return None
        clip_id, modality = parts[1], parts[2]
    else:
        return None

    timestamp_match = _TIMESTAMP_RE.search(path.name)
    frame_match = _FRAME_RE.search(path.name)
    return ParsedMember(
        split=source.split,
        clip_id=clip_id,
        modality=modality,
        member_path=member.path,
        size=member.size,
        action_id=action_id,
        action_name=action_name,
        user=user,
        trial=trial,
        timestamp=timestamp_match.group(1) if timestamp_match else None,
        frame_index=int(frame_match.group(1)) if frame_match else None,
    )


def _new_record(source: DataSource, parsed: ParsedMember) -> dict[str, object]:
    record: dict[str, object] = {
        "split": parsed.split,
        "clip_id": parsed.clip_id,
        "action_id": parsed.action_id,
        "action_name": parsed.action_name,
        "user": parsed.user,
        "trial": parsed.trial,
        "source_kind": source.kind,
        "source_path": source.path,
    }
    for modality in MODALITIES:
        slug = MODALITY_SLUGS[modality]
        record[f"{slug}_present"] = False
        record[f"{slug}_file_count"] = 0
        record[f"{slug}_bytes"] = 0
    return record


def _read_from_directory(source: DataSource, member_path: str) -> bytes:
    root = Path(source.path)
    parts = PurePosixPath(member_path).parts
    if not parts or parts[0] != root.name:
        raise ValueError(f"Member is outside source root: {member_path}")
    return root.joinpath(*parts[1:]).read_bytes()


@contextmanager
def open_source_reader(source: DataSource) -> Iterator[Callable[[str], bytes]]:
    """Yield a member reader, keeping the backing archive open throughout.

    Callers that process many members should read them one at a time through
    this reader rather than materializing every payload at once.
    """

    if not source.readable:
        raise RuntimeError("Multipart ZIP payloads must be merged or extracted before reading.")
    if source.kind == "directory":
        yield lambda member_path: _read_from_directory(source, member_path)
    else:
        with zipfile.ZipFile(source.path) as archive:
            yield archive.read


def read_members(source: DataSource, member_paths: Sequence[str]) -> dict[str, bytes]:
    """Read several members while opening the backing archive only once.

    Every payload is held in memory, so this suits bounded sets such as the
    frames of a single clip. Use ``open_source_reader`` for whole-split scans.
    """

    with open_source_reader(source) as read_member:
        return {path: read_member(path) for path in member_paths}


def _csv_data_row_count(data: bytes) -> int:
    text = data.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(text.splitlines()))
    return max(0, len(rows) - 1)


def _enrich_content_quality(
    source: DataSource,
    records: Mapping[str, dict[str, object]],
    members_by_clip: Mapping[str, Mapping[str, list[ParsedMember]]],
) -> None:
    if not source.readable:
        return

    with open_source_reader(source) as read_member:
        for clip_id, modalities in members_by_clip.items():
            record = records[clip_id]
            imu_empty = 0
            for item in modalities.get("IMU", []):
                if _csv_data_row_count(read_member(item.member_path)) == 0:
                    imu_empty += 1
            record["imu_empty_files"] = imu_empty

            radar_items = modalities.get("Radar", [])
            if radar_items:
                radar_rows = sum(
                    _csv_data_row_count(read_member(item.member_path)) for item in radar_items
                )
                record["radar_data_rows"] = radar_rows
                record["radar_empty"] = radar_rows == 0

            skeleton_empty = 0
            skeleton_multi = 0
            skeleton_bad = 0
            for item in modalities.get("Skeleton", []):
                try:
                    people = json.loads(read_member(item.member_path))
                    if isinstance(people, list):
                        skeleton_empty += int(len(people) == 0)
                        skeleton_multi += int(len(people) > 1)
                    else:
                        skeleton_bad += 1
                except (UnicodeDecodeError, json.JSONDecodeError):
                    skeleton_bad += 1
            record["skeleton_empty_frames"] = skeleton_empty
            record["skeleton_multi_person_frames"] = skeleton_multi
            record["skeleton_bad_json"] = skeleton_bad


def _natural_key(value: object) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", "" if value is None else str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _build_manifest_from_members(
    source: DataSource,
    members: Iterable[Member],
    *,
    deep: bool = False,
) -> list[dict[str, object]]:
    """Build compact clip records from a bounded or complete member stream."""

    records: dict[str, dict[str, object]] = {}
    members_by_clip: dict[str, dict[str, list[ParsedMember]]] = defaultdict(lambda: defaultdict(list))
    timestamps: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for member in members:
        parsed = parse_member(source, member)
        if parsed is None:
            continue
        record = records.setdefault(parsed.clip_id, _new_record(source, parsed))
        slug = MODALITY_SLUGS[parsed.modality]
        record[f"{slug}_present"] = True
        record[f"{slug}_file_count"] = int(record[f"{slug}_file_count"]) + 1
        if parsed.size is not None:
            record[f"{slug}_bytes"] = int(record[f"{slug}_bytes"]) + parsed.size
        members_by_clip[parsed.clip_id][parsed.modality].append(parsed)
        if parsed.timestamp:
            timestamps[parsed.clip_id][parsed.modality].add(parsed.timestamp)

    if deep:
        _enrich_content_quality(source, records, members_by_clip)

    for clip_id, record in records.items():
        modality_count = sum(bool(record[f"{MODALITY_SLUGS[m]}_present"]) for m in MODALITIES)
        missing = [m for m in MODALITIES if not record[f"{MODALITY_SLUGS[m]}_present"]]
        record["modality_count"] = modality_count
        record["complete"] = modality_count == len(MODALITIES)
        record["missing_modalities"] = ", ".join(missing)
        record["total_file_count"] = sum(int(record[f"{MODALITY_SLUGS[m]}_file_count"]) for m in MODALITIES)

        clip_times = timestamps.get(clip_id, {})
        preferred_times = clip_times.get("Depth_Color") or clip_times.get("IR") or set()
        if preferred_times:
            start = min(preferred_times)
            end = max(preferred_times)
            start_dt = datetime.strptime(start, "%Y-%m-%d_%H-%M-%S.%f")
            end_dt = datetime.strptime(end, "%Y-%m-%d_%H-%M-%S.%f")
            record["start_timestamp"] = start
            record["end_timestamp"] = end
            record["duration_seconds"] = (end_dt - start_dt).total_seconds()

        depth = clip_times.get("Depth_Color")
        infrared = clip_times.get("IR")
        skeleton = clip_times.get("Skeleton")
        if depth is not None and infrared is not None:
            record["depth_ir_aligned"] = depth == infrared
        if depth is not None and skeleton is not None:
            record["depth_skeleton_aligned"] = depth == skeleton

        issues: list[str] = []
        if missing:
            issues.append(f"missing: {', '.join(missing)}")
        if record.get("radar_empty") is True:
            issues.append("radar has no detections")
        if int(record.get("imu_empty_files", 0) or 0) > 0:
            issues.append(f"{record['imu_empty_files']} empty IMU file(s)")
        if int(record.get("skeleton_bad_json", 0) or 0) > 0:
            issues.append(f"{record['skeleton_bad_json']} invalid skeleton JSON file(s)")
        if record.get("depth_ir_aligned") is False:
            issues.append("Depth/IR timestamps differ")
        if record.get("depth_skeleton_aligned") is False:
            issues.append("Depth/Skeleton timestamps differ")
        record["issues"] = "; ".join(issues)

    return sorted(
        records.values(),
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            row["action_id"] if row["action_id"] is not None else 10_000,
            _natural_key(row["user"]),
            _natural_key(row["trial"]),
            _natural_key(row["clip_id"]),
        ),
    )


def build_source_manifest(source: DataSource, deep: bool = False) -> list[dict[str, object]]:
    """Build one compact, scalar-only record per logical clip."""

    return _build_manifest_from_members(source, iter_source_members(source), deep=deep)


def count_source_members(source: DataSource) -> int:
    """Count physical files efficiently for background progress reporting."""

    if source.kind == "directory":
        total = 0
        pending = [source.path]
        while pending:
            current = pending.pop()
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total += 1
        return total
    return sum(1 for _ in iter_source_members(source))


def _members_with_progress(
    source: DataSource,
    callback: Callable[[int], None],
) -> Iterator[Member]:
    processed = 0
    for processed, member in enumerate(iter_source_members(source), start=1):
        yield member
        if processed % 1_000 == 0:
            callback(processed)
    callback(processed)


def _directory_clip_ids(source: DataSource) -> list[str]:
    """Discover logical clip IDs using directory names rather than file scans."""

    if source.kind != "directory":
        raise ValueError("Fast initial clip discovery requires an extracted directory source")

    root = Path(source.path)
    if source.split == "test":
        return sorted(
            (
                path.name
                for path in root.iterdir()
                if path.is_dir() and path.name.startswith("SM_test_")
            ),
            key=_natural_key,
        )

    clips_by_action: dict[str, list[str]] = defaultdict(list)
    # One visual modality is enough to discover a representative initial set.
    # Try the modalities in their normal display order in case Depth_Color is
    # absent from a particular extracted copy.
    for modality in MODALITIES:
        modality_root = root / "data" / modality
        if not modality_root.is_dir():
            continue
        action_dirs = sorted(
            (path for path in modality_root.iterdir() if path.is_dir()),
            key=lambda path: _natural_key(path.name),
        )
        for action_dir in action_dirs:
            user_dirs = sorted(
                (path for path in action_dir.iterdir() if path.is_dir()),
                key=lambda path: _natural_key(path.name),
            )
            for user_dir in user_dirs:
                trial_dirs = sorted(
                    (path for path in user_dir.iterdir() if path.is_dir()),
                    key=lambda path: _natural_key(path.name),
                )
                for trial_dir in trial_dirs:
                    clips_by_action[action_dir.name].append(
                        f"{action_dir.name}/{user_dir.name}/{trial_dir.name}"
                    )
        if clips_by_action:
            break

    # Round-robin across actions so the initial training view is useful instead
    # of containing only the first few action directories.
    ordered: list[str] = []
    action_names = sorted(clips_by_action, key=_natural_key)
    offset = 0
    while True:
        added = False
        for action_name in action_names:
            clips = clips_by_action[action_name]
            if offset < len(clips):
                ordered.append(clips[offset])
                added = True
        if not added:
            break
        offset += 1
    return ordered


def initial_clip_ids(source: DataSource, limit: int) -> list[str]:
    """Choose a small, representative set of clips for progressive startup."""

    if limit <= 0:
        return []
    clip_ids = _directory_clip_ids(source)
    if len(clip_ids) <= limit:
        return clip_ids
    if source.split == "train":
        return clip_ids[:limit]
    if limit == 1:
        return [clip_ids[0]]
    # Spread test clips over the full ID range rather than showing only the
    # lowest-numbered anonymous samples.
    indices = [round(index * (len(clip_ids) - 1) / (limit - 1)) for index in range(limit)]
    return [clip_ids[index] for index in indices]


def _iter_directory_clip_members(source: DataSource, clip_ids: Iterable[str]) -> Iterator[Member]:
    root = Path(source.path)
    relative_root = root.parent
    for clip_id in clip_ids:
        for modality in MODALITIES:
            clip_dir = (
                root / "data" / modality / clip_id
                if source.split == "train"
                else root / clip_id / modality
            )
            if not clip_dir.is_dir():
                continue
            for path in clip_dir.rglob("*"):
                if path.is_file():
                    yield Member(path.relative_to(relative_root).as_posix(), path.stat().st_size)


def build_initial_dataset_manifest(
    dataset_root: str | Path,
    *,
    max_clips: int = 200,
) -> tuple[list[dict[str, object]], dict[str, DataSource]]:
    """Build a representative partial manifest without a full file scan.

    The budget is divided evenly between available train and test directory
    sources. Archive sources are omitted because finding complete clips inside
    an archive requires scanning its full central directory.
    """

    sources = discover_sources(dataset_root)
    progressive_sources = {
        split: source for split, source in sources.items() if source.kind == "directory"
    }
    if not progressive_sources or max_clips <= 0:
        return [], sources

    split_names = [split for split in ("train", "test") if split in progressive_sources]
    base, remainder = divmod(max_clips, len(split_names))
    selected: dict[str, list[str]] = {}
    for index, split in enumerate(split_names):
        quota = base + int(index < remainder)
        selected[split] = initial_clip_ids(progressive_sources[split], quota)

    # If a small split cannot fill its quota, let the other split use the
    # remaining budget.
    remaining = max_clips - sum(len(values) for values in selected.values())
    if remaining > 0:
        for split in split_names:
            existing = selected[split]
            candidates = initial_clip_ids(progressive_sources[split], len(existing) + remaining)
            existing_set = set(existing)
            additions = [clip_id for clip_id in candidates if clip_id not in existing_set]
            selected[split].extend(additions[:remaining])
            remaining = max_clips - sum(len(values) for values in selected.values())
            if remaining == 0:
                break

    records: list[dict[str, object]] = []
    for split in split_names:
        source = progressive_sources[split]
        members = _iter_directory_clip_members(source, selected[split])
        records.extend(_build_manifest_from_members(source, members, deep=False))
    return records, sources


def build_dataset_manifest(
    dataset_root: str | Path,
    *,
    deep_test: bool = True,
    deep_train: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, DataSource]]:
    """Discover sources and build a combined train/test manifest."""

    sources = discover_sources(dataset_root)
    source_counts: dict[str, int] = {}
    if progress_callback is not None:
        progress_callback("counting", 0, 0)
        source_counts = {split: count_source_members(source) for split, source in sources.items()}
        progress_callback("scanning", 0, sum(source_counts.values()))

    records: list[dict[str, object]] = []
    processed_before = 0
    for split in ("train", "test"):
        if split not in sources:
            continue
        source = sources[split]
        if progress_callback is None:
            members = iter_source_members(source)
        else:
            total = sum(source_counts.values())
            members = _members_with_progress(
                source,
                lambda processed, split=split, before=processed_before: progress_callback(
                    split, before + processed, total
                ),
            )
        records.extend(
            _build_manifest_from_members(
                source,
                members,
                deep=deep_train if split == "train" else deep_test,
            )
        )
        if progress_callback is not None:
            processed_before += source_counts[split]
            progress_callback(split, processed_before, sum(source_counts.values()))
    return records, sources


def _member_sort_key(item: ParsedMember) -> tuple[str, int, str]:
    """Playback order for members of one modality: timestamp, frame, path."""

    return (
        item.timestamp or "",
        item.frame_index if item.frame_index is not None else -1,
        item.member_path,
    )


def _sorted_member_paths(parsed_by_modality: Mapping[str, list[ParsedMember]]) -> dict[str, list[str]]:
    """Collapse parsed members into sorted member-path lists per modality."""

    return {
        modality: [item.member_path for item in sorted(items, key=_member_sort_key)]
        for modality, items in parsed_by_modality.items()
    }


def build_member_index(source: DataSource) -> dict[str, dict[str, list[str]]]:
    """Return sorted member paths grouped by logical clip and modality."""

    index: dict[str, dict[str, list[ParsedMember]]] = defaultdict(lambda: defaultdict(list))
    for member in iter_source_members(source):
        parsed = parse_member(source, member)
        if parsed is not None:
            index[parsed.clip_id][parsed.modality].append(parsed)

    return {clip_id: _sorted_member_paths(modalities) for clip_id, modalities in index.items()}


def build_clip_member_index(source: DataSource, clip_id: str) -> dict[str, list[str]]:
    """Return sorted member paths for a single clip (avoids full-dataset scan).

    The clip explorer previously built an index for all 3k+ clips (19.5 s for
    train, 2.1 s for test on the reference workstation) even though only one
    clip is displayed. This helper scans only the directories/files that can
    belong to ``clip_id``, reducing the first-clip latency to ~0.02 s.
    """

    parsed_by_modality: dict[str, list[ParsedMember]] = defaultdict(list)

    # Directory sources can be listed directly without a full rglob.
    if source.kind == "directory":
        root = Path(source.path)
        relative_root = root.parent
        for modality in MODALITIES:
            # train clip_id = "<action>/<user>/<trial>", test clip_id = "SM_test_XXXX"
            clip_dir = root / "data" / modality / clip_id if source.split == "train" else root / clip_id / modality
            if not clip_dir.is_dir():
                continue
            for path in clip_dir.rglob("*"):
                if not path.is_file():
                    continue
                member = Member(path.relative_to(relative_root).as_posix(), path.stat().st_size)
                parsed = parse_member(source, member)
                if parsed is not None and parsed.clip_id == clip_id:
                    parsed_by_modality[modality].append(parsed)
        return _sorted_member_paths(parsed_by_modality)

    # Zip / multipart: filtering the central directory is cheap (no file I/O).
    for member in iter_source_members(source):
        # Quick substring pre-filter before the more expensive parse_member.
        if clip_id not in member.path:
            continue
        parsed = parse_member(source, member)
        if parsed is not None and parsed.clip_id == clip_id:
            parsed_by_modality[parsed.modality].append(parsed)
    return _sorted_member_paths(parsed_by_modality)


def write_manifest(records: Iterable[Mapping[str, object]], output_path: str | Path) -> Path:
    """Atomically write records as Parquet, CSV, or JSON."""

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}")
    rows = list(records)
    suffix = output.suffix.lower()
    try:
        if suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError("Parquet output requires the dashboard dependencies.") from exc
            pd.DataFrame(rows).to_parquet(temporary, index=False)
        elif suffix == ".csv":
            fields = sorted({field for row in rows for field in row})
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        elif suffix == ".json":
            temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            raise ValueError("Manifest output must end in .parquet, .csv, or .json")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
