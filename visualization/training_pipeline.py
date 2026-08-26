"""Streamlit page and process helpers for running the training pipeline."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Mapping

from visualization.progress import COMPLETE, ERROR, RUNNING


RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PIPELINE_STAGES = (
    ("discovery", "Discover dataset"),
    ("features", "Extract or load features"),
    ("validation", "Validate parameters"),
    ("training", "Fit final model"),
    ("saving", "Save reusable outputs"),
)
STAGE_INDEX = {name: index for index, (name, _label) in enumerate(PIPELINE_STAGES)}
FULL_DATASET = "Full dataset"
SAMPLE_DATASET = "Sample dataset"
DATASET_CHOICES = (FULL_DATASET, SAMPLE_DATASET)


@dataclass(frozen=True)
class TrainingPipelineConfig:
    dataset_root: Path
    test_csv: Path
    algorithm: str
    run_name: str
    feature_cache: Path
    artifacts_dir: Path
    output_path: Path
    n_jobs: int = 4
    folds: int = 5
    cv_repeats: int = 1
    selection_metric: str = "macro_f1"
    random_state: int = 42
    search_space: Mapping[str, list[Any]] | None = None
    parameters: Mapping[str, Any] | None = None
    rebuild_features: bool = False
    extract_only: bool = False
    skip_validation: bool = False
    model_output_path: Path | None = None


@dataclass
class TrainingProcess:
    process: subprocess.Popen[str]
    config: TrainingPipelineConfig
    run_dir: Path
    log_path: Path
    progress_path: Path
    metadata_path: Path
    log_handle: IO[str]
    started_at: str
    repository_root: Path
    finalized: bool = False
    cancel_requested: bool = False
    return_code: int | None = None
    outputs_applied: bool = False

    @property
    def display_log_path(self) -> str:
        """Repository-relative log path, or the absolute one if it moved."""

        try:
            return str(self.log_path.relative_to(self.repository_root))
        except ValueError:
            return str(self.log_path)


def workflow_dataset_paths(
    repository_root: Path,
    default_dataset_root: str | Path,
    state: Mapping[str, object],
) -> tuple[Path, Path, bool]:
    """Resolve the dataset and test CSV selected by the shared workflow UI."""

    sample = state.get("workflow_dataset_choice") == SAMPLE_DATASET
    dataset_root = (
        repository_root / "artifacts" / "sample_dataset"
        if sample
        else Path(
            str(state.get("workflow_full_dataset_root") or default_dataset_root)
        ).expanduser()
    )
    default_test_csv = (
        dataset_root / "Testing" / "test.csv"
        if sample
        else repository_root / "Testing" / "test.csv"
    )
    test_csv = Path(
        str(state.get("workflow_test_csv") or default_test_csv)
    ).expanduser()
    return dataset_root.resolve(), test_csv.resolve(), sample


def parse_json_object(value: str, label: str) -> dict[str, Any] | None:
    """Parse an optional JSON object used by a pipeline form field."""

    if not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def repository_output_path(
    repository_root: Path, value: str | Path, label: str
) -> Path:
    """Resolve an output and require it to remain inside the repository."""

    root = repository_root.expanduser().resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository: {root}") from exc
    if resolved == root:
        raise ValueError(
            f"{label} must name a file or subdirectory inside the repository"
        )
    return resolved


def model_output_path(repository_root: Path, value: str | Path) -> Path:
    """Resolve a model file and require its own ``artifacts/<run>/`` directory.

    The trainer writes ``validation.json`` and ``oof_predictions.npz`` beside
    the model, and ``discover_validation_reports`` only globs
    ``artifacts/*/validation.json``. A shallower path such as ``model.joblib``
    passes the repository check but would drop those reports in the checkout
    root, where nothing finds them; anything deeper hides them from the same
    glob. Pinning the depth keeps every trained run visible on the leaderboard.
    """

    resolved = repository_output_path(repository_root, value, "Model output")
    if resolved.suffix.lower() != ".joblib":
        raise ValueError("Model output must use the .joblib extension")
    artifacts_root = (repository_root.expanduser().resolve()) / "artifacts"
    if resolved.parent.parent != artifacts_root:
        raise ValueError(
            "Model output must be artifacts/<run-name>/<file>.joblib so its "
            "validation report is discovered by Algorithm comparison."
        )
    return resolved



def reject_submission_name(prediction_split: str, output: Path) -> None:
    """Refuse to write a non-gradeable file over a submission name.

    Only a test-only run produces the ``path,prediction`` competition format.
    Train-only and combined files are keyed by ``clip_id`` and carry training
    rows, so writing one to ``*_submission.csv`` would replace a gradeable file
    with one the grader rejects.
    """

    if prediction_split == "Test only" or not output.name.endswith("_submission.csv"):
        return
    raise ValueError(
        f"'{prediction_split}' predictions are keyed by clip_id and include training "
        "rows, so they are not a valid submission. Choose a name that does not end "
        "in _submission.csv."
    )


def validate_run_name(value: str) -> str:
    name = value.strip()
    if not RUN_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Run name must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens."
        )
    return name


def next_available_run_name(repository_root: Path, base_name: str) -> str:
    """Choose a repo-local artifact/output name that does not overwrite a run."""

    root = repository_root.resolve()
    candidate = validate_run_name(base_name)
    suffix = 1
    while (root / "artifacts" / candidate).exists() or (
        root / "outputs" / f"{candidate}_submission.csv"
    ).exists():
        suffix += 1
        candidate = f"{base_name}-{suffix}"
    return candidate


def build_training_command(
    config: TrainingPipelineConfig,
    progress_path: Path,
) -> list[str]:
    """Translate a validated UI configuration into the existing CLI contract."""

    command = [
        sys.executable,
        "-u",
        "-m",
        "modeling.train",
        "--algorithm",
        config.algorithm,
        "--dataset-root",
        str(config.dataset_root),
        "--test-csv",
        str(config.test_csv),
        "--features-cache",
        str(config.feature_cache),
        "--artifacts-dir",
        str(config.artifacts_dir),
        "--output",
        str(config.output_path),
        "--n-jobs",
        str(config.n_jobs),
        "--folds",
        str(config.folds),
        "--cv-repeats",
        str(config.cv_repeats),
        "--selection-metric",
        config.selection_metric,
        "--random-state",
        str(config.random_state),
        "--progress-file",
        str(progress_path),
    ]
    if config.search_space is not None and not config.skip_validation:
        command.extend(
            ("--search-space", json.dumps(config.search_space, sort_keys=True))
        )
    if config.parameters is not None and config.skip_validation:
        command.extend(("--parameters", json.dumps(config.parameters, sort_keys=True)))
    if config.rebuild_features:
        command.append("--rebuild-features")
    if config.extract_only:
        command.append("--extract-only")
    if config.skip_validation:
        command.append("--skip-validation")
    if config.model_output_path is not None:
        command.extend(("--model-output", str(config.model_output_path)))
    return command


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _serializable_config(config: TrainingPipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "dataset_root",
        "test_csv",
        "feature_cache",
        "artifacts_dir",
        "output_path",
    ):
        payload[key] = str(payload[key])
    if payload["model_output_path"] is not None:
        payload["model_output_path"] = str(payload["model_output_path"])
    return payload


def start_training_run(
    repository_root: Path,
    config: TrainingPipelineConfig,
) -> TrainingProcess:
    """Start one unbuffered training process and persist its command and logs."""

    root = repository_root.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "artifacts" / "training_runs" / f"{timestamp}-{config.run_name}"
    counter = 1
    while run_dir.exists():
        counter += 1
        run_dir = (
            root
            / "artifacts"
            / "training_runs"
            / (f"{timestamp}-{config.run_name}-{counter}")
        )
    run_dir.mkdir(parents=True)
    log_path = run_dir / "training.log"
    progress_path = run_dir / "progress.json"
    metadata_path = run_dir / "run.json"
    command = build_training_command(config, progress_path)
    started_at = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "command": command,
        "config": _serializable_config(config),
        "log": str(log_path),
        "progress": str(progress_path),
    }
    _write_json(metadata_path, metadata)

    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    log_handle.write(f"$ {shlex.join(command)}\n\n")
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
        )
    except Exception:
        log_handle.close()
        metadata.update(
            {
                "status": "failed_to_start",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(metadata_path, metadata)
        raise
    metadata["pid"] = process.pid
    _write_json(metadata_path, metadata)
    return TrainingProcess(
        process=process,
        config=config,
        run_dir=run_dir,
        log_path=log_path,
        progress_path=progress_path,
        metadata_path=metadata_path,
        log_handle=log_handle,
        started_at=started_at,
        repository_root=root,
    )


def read_training_progress(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_log_tail(path: Path, max_characters: int = 24_000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_characters))
            text = handle.read()
    except OSError:
        return ""
    if size > max_characters:
        return "… earlier log output omitted …\n" + text
    return text


def finalize_training_run(run: TrainingProcess) -> int | None:
    """Poll a run, close its log once, and persist its terminal status."""

    return_code = run.process.poll()
    if return_code is None or run.finalized:
        return return_code
    run.log_handle.close()
    run.finalized = True
    run.return_code = return_code
    try:
        metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    metadata.update(
        {
            "status": (
                "cancelled"
                if run.cancel_requested
                else "completed" if return_code == 0 else "failed"
            ),
            "return_code": return_code,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(run.metadata_path, metadata)
    return return_code


def cancel_training_run(run: TrainingProcess) -> None:
    """Request termination of the process tree created for a UI run."""

    if run.process.poll() is not None:
        return
    run.cancel_requested = True
    terminate_process_group(run.process.pid)


def terminate_process_group(pid: int) -> None:
    """Signal a training process started with its own session."""

    if os.name != "nt":
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    else:  # pragma: no cover - Windows is not used by the project CI
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def process_is_alive(pid: int | None) -> bool:
    """Best-effort liveness check for a run this session did not start."""

    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user, so it exists but is not ours to signal.
        return True
    except OSError:
        return False
    return True


def reconcile_orphaned_runs(repository_root: Path) -> list[dict[str, Any]]:
    """Repair run records the session that started them never finalized.

    A run outlives the browser session that launched it: the child is started
    in its own session and the ``TrainingProcess`` lives only in
    ``st.session_state``. Without this, ``run.json`` claims ``running`` forever
    and a still-live run is invisible and uncancellable from a new session.
    Returns the runs that are genuinely still executing.
    """

    runs_root = repository_root / "artifacts" / "training_runs"
    if not runs_root.is_dir():
        return []
    alive: list[dict[str, Any]] = []
    for metadata_path in sorted(runs_root.glob("*/run.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict) or metadata.get("status") != "running":
            continue
        pid = metadata.get("pid")
        if process_is_alive(pid):
            metadata["run_dir"] = str(metadata_path.parent)
            alive.append(metadata)
            continue
        metadata.update(
            {
                "status": "interrupted",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "note": "The dashboard session ended before this run was finalized.",
            }
        )
        try:
            _write_json(metadata_path, metadata)
        except OSError:
            continue
    return alive


def overall_progress(progress: Mapping[str, Any], return_code: int | None) -> float:
    if return_code == 0 or (
        return_code is None and progress.get("status") == COMPLETE
    ):
        return 1.0
    stage = str(progress.get("stage", "discovery"))
    ranges = {
        "discovery": (0.01, 0.08),
        "features": (0.08, 0.48),
        "validation": (0.48, 0.82),
        "training": (0.82, 0.94),
        "saving": (0.94, 0.99),
    }
    start, end = ranges.get(stage, (0.01, 0.02))
    current = progress.get("current")
    total = progress.get("total")
    if (
        isinstance(current, (int, float))
        and isinstance(total, (int, float))
        and total > 0
    ):
        return min(end, start + (end - start) * max(0.0, min(1.0, current / total)))
    return start


def stage_states(
    progress: Mapping[str, Any],
    config: TrainingPipelineConfig,
    return_code: int | None,
) -> list[tuple[str, str]]:
    """Return display labels paired with completed/active/pending/skipped."""

    current_stage = str(progress.get("stage", "discovery"))
    current_index = STAGE_INDEX.get(current_stage, 0)
    status = str(progress.get("status", RUNNING))
    completed = return_code == 0 or (return_code is None and status == COMPLETE)
    reported_error = status == ERROR
    rows: list[tuple[str, str]] = []
    for index, (stage, label) in enumerate(PIPELINE_STAGES):
        skipped = (
            config.extract_only and stage in {"validation", "training", "saving"}
        ) or (config.skip_validation and stage == "validation")
        if skipped:
            state = "skipped"
        elif completed or index < current_index:
            state = "completed"
        elif (
            (return_code is not None and return_code != 0) or reported_error
        ) and index == current_index:
            state = "failed"
        elif index == current_index:
            state = "active"
        else:
            state = "pending"
        rows.append((label, state))
    return rows


def adopt_run_outputs(run: TrainingProcess) -> dict[str, str]:
    """Return the session-state entries a finished run should hand over, once.

    The monitor keeps rendering for as long as a finished run is on screen, so
    applying these on every pass would overwrite whatever the user selected in
    the Prediction step afterwards. Returns an empty mapping on later calls.
    """

    if run.outputs_applied:
        return {}
    run.outputs_applied = True
    updates: dict[str, str] = {}
    if run.config.output_path.is_file():
        updates["prediction_csv_input"] = str(run.config.output_path)
    model_path = run.config.model_output_path or run.config.artifacts_dir / "model.joblib"
    if model_path.is_file():
        updates["model_artifact_input"] = str(model_path)
        updates["workflow_model_path"] = str(model_path)
    return updates


def _render_run_monitor(run: TrainingProcess) -> None:
    import streamlit as st

    # Poll only while the process is alive. A fragment with a fixed interval
    # keeps ticking for as long as the page is open, re-reading the log and
    # re-assigning the sidebar's session state once a second after the run has
    # already finished.
    poll_interval = 1.0 if run.process.poll() is None else None

    @st.fragment(run_every=poll_interval)
    def monitor() -> None:
        return_code = finalize_training_run(run)
        if poll_interval is not None and return_code is not None:
            # The run finished during a polling tick. Rerun the page so the
            # fragment is rebuilt with polling switched off; otherwise it keeps
            # ticking against a process that has already exited.
            st.rerun()
        progress = read_training_progress(run.progress_path)
        message = str(progress.get("message", "Starting training process…"))
        st.progress(overall_progress(progress, return_code), text=message)

        icons = {
            "completed": "✅",
            "active": "🔄",
            "pending": "○",
            "skipped": "—",
            "failed": "❌",
        }
        columns = st.columns(len(PIPELINE_STAGES))
        for column, (label, state) in zip(
            columns, stage_states(progress, run.config, return_code), strict=True
        ):
            column.caption(f"{icons[state]} {label}")

        with st.expander("Live training log", expanded=return_code not in (0,)):
            log_text = read_log_tail(run.log_path)
            st.code(log_text or "Waiting for output…", language="text")
            st.caption(f"Persistent log: `{run.display_log_path}`")

        if return_code is None:
            if run.cancel_requested:
                st.warning("Stopping the training process…")
            elif st.button("Stop training", key=f"stop_training_{run.run_dir.name}"):
                cancel_training_run(run)
                st.rerun(scope="fragment")
            return

        if return_code == 0:
            if run.config.extract_only:
                st.success(
                    "Feature extraction completed. The cache is ready for later runs."
                )
            else:
                st.success(
                    "Training pipeline completed. Its outputs are ready for the other pages."
                )
            outputs = progress.get("outputs", {})
            if isinstance(outputs, dict):
                for label, path in outputs.items():
                    st.caption(f"{label.replace('_', ' ').title()}: `{path}`")
            st.session_state.update(adopt_run_outputs(run))
        elif run.cancel_requested:
            st.warning(
                "Training was stopped. The log and any files already written were retained."
            )
        else:
            st.error(
                f"Training failed with exit status {return_code}. See the log above for details."
            )

    monitor()


def _render_orphan_run(metadata: Mapping[str, Any], repository_root: Path) -> None:
    """Monitor a run still executing from an earlier dashboard session."""

    import streamlit as st

    run_dir = Path(str(metadata.get("run_dir")))
    config = metadata.get("config") or {}
    name = str(config.get("run_name") or run_dir.name)
    pid = metadata.get("pid")

    st.warning(
        f"A training run started in an earlier session is still going: **{name}**. "
        "Its outputs land in the usual places; this session can watch it and stop it, "
        "but cannot recover the original form."
    )

    @st.fragment(run_every=2.0)
    def orphan_monitor() -> None:
        if not process_is_alive(pid):
            reconcile_orphaned_runs(repository_root)
            st.info(f"Run {name} has finished. Reload the page for its outputs.")
            return
        progress = read_training_progress(run_dir / "progress.json")
        message = str(progress.get("message", "Running…"))
        st.progress(overall_progress(progress, None), text=f"{name}: {message}")
        with st.expander("Live training log", expanded=False):
            st.code(
                read_log_tail(run_dir / "training.log") or "Waiting for output…",
                language="text",
            )
        if st.button("Stop this run", key=f"stop_orphan_{run_dir.name}"):
            terminate_process_group(int(pid))
            st.rerun(scope="fragment")

    orphan_monitor()


def _saved_outputs(repository_root: Path) -> list[dict[str, object]]:
    artifacts_root = repository_root / "artifacts"
    outputs_root = repository_root / "outputs"
    names = {path.parent.name for path in artifacts_root.glob("*/*.joblib")} | {
        path.parent.name for path in artifacts_root.glob("*/validation.json")
    }
    rows: list[dict[str, object]] = []
    for name in sorted(names):
        artifact_dir = artifacts_root / name
        models = sorted(
            artifact_dir.glob("*.joblib"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        model = models[0] if models else artifact_dir / "model.joblib"
        report = artifact_dir / "validation.json"
        submission = outputs_root / f"{name}_submission.csv"
        modified = 0.0
        for path in (model, report, submission):
            try:
                modified = max(modified, path.stat().st_mtime)
            except OSError:
                # Removed between the glob and the stat, or unreadable.
                continue
        rows.append(
            {
                "run": name,
                "model": (
                    str(model.relative_to(repository_root)) if model.is_file() else ""
                ),
                "validation": (
                    str(report.relative_to(repository_root)) if report.is_file() else ""
                ),
                "submission": (
                    str(submission.relative_to(repository_root))
                    if submission.is_file()
                    else ""
                ),
                "modified": (
                    datetime.fromtimestamp(modified, timezone.utc).isoformat(
                        timespec="seconds"
                    )
                    if modified
                    else ""
                ),
            }
        )
    rows.sort(key=lambda row: str(row["modified"]), reverse=True)
    return rows


def render_training_pipeline(repository_root: Path, default_dataset_root: str) -> None:
    """Render one coherent dataset → features → training → prediction workflow."""

    import streamlit as st

    from visualization.predictions import (
        combine_prediction_tables,
        discover_model_artifacts,
        generate_all_split_predictions,
        generate_predictions_from_model,
        save_prediction_csv,
    )

    st.header("Model workflow")
    st.caption(
        "Choose the dataset once, then carry the same files through feature "
        "extraction, training, prediction, and visualization."
    )

    try:
        from modeling.algorithms import available_algorithms, get_algorithm

        algorithm_names = available_algorithms()
        algorithms = {name: get_algorithm(name) for name in algorithm_names}
    except ImportError as exc:
        st.error(
            "Training dependencies are not installed. Install `modeling/requirements.txt` "
            f"and restart the app. ({exc})"
        )
        return

    if not algorithm_names:
        st.error("No training algorithms are registered.")
        return

    default_algorithm = (
        "logistic_regression"
        if "logistic_regression" in algorithm_names
        else algorithm_names[0]
    )
    sample_root = repository_root / "artifacts" / "sample_dataset"

    if st.session_state.get("workflow_dataset_choice") not in DATASET_CHOICES:
        # Repairing the choice bypasses the radio's on_change, so drop any test
        # CSV that belonged to the mode being replaced.
        st.session_state["workflow_dataset_choice"] = FULL_DATASET
        st.session_state.pop("workflow_test_csv", None)
    st.session_state.setdefault("workflow_full_dataset_root", str(default_dataset_root))
    if st.session_state.get("workflow_algorithm") not in algorithms:
        st.session_state["workflow_algorithm"] = default_algorithm

    def default_test_csv(sample: bool) -> Path:
        return (
            sample_root / "Testing" / "test.csv"
            if sample
            else repository_root / "Testing" / "test.csv"
        )

    def reset_algorithm_outputs() -> None:
        selected = algorithms[str(st.session_state["workflow_algorithm"])]
        sample = st.session_state.get("workflow_dataset_choice") == SAMPLE_DATASET
        base_name = (
            f"sample-{selected.artifact_name}" if sample else selected.artifact_name
        )
        run_name = next_available_run_name(repository_root, base_name)
        st.session_state["workflow_run_name"] = run_name
        st.session_state["workflow_model_output"] = (
            f"artifacts/{run_name}/model.joblib"
        )
        st.session_state["workflow_training_output"] = (
            f"outputs/{run_name}_submission.csv"
        )
        st.session_state["workflow_model_path"] = (
            f"artifacts/{run_name}/model.joblib"
        )
        st.session_state["workflow_prediction_output"] = (
            f"outputs/{run_name}_predictions.csv"
        )
        search_space = (
            {key: [value] for key, value in selected.default_parameters.items()}
            if sample
            else {
                key: list(values)
                for key, values in selected.default_search_space.items()
            }
        )
        st.session_state["workflow_search_space"] = json.dumps(
            search_space, indent=2, sort_keys=True
        )
        st.session_state["workflow_parameters"] = json.dumps(
            dict(selected.default_parameters), indent=2, sort_keys=True
        )
        st.session_state["workflow_folds"] = 2 if sample else 5
        st.session_state["workflow_repeats"] = 1

    def dataset_changed() -> None:
        sample = st.session_state.get("workflow_dataset_choice") == SAMPLE_DATASET
        st.session_state["workflow_test_csv"] = str(default_test_csv(sample))
        st.session_state["workflow_feature_cache"] = (
            "artifacts/features/ui_sample.npz"
            if sample
            else "artifacts/features/four_sensor_v3.npz"
        )
        st.session_state["saved_manifest_input"] = ""
        st.session_state["use_manifest_checkbox"] = False
        st.session_state["prediction_csv_input"] = ""
        reset_algorithm_outputs()

    def full_root_changed() -> None:
        if st.session_state.get("workflow_dataset_choice") != FULL_DATASET:
            return
        st.session_state["workflow_test_csv"] = str(default_test_csv(False))
        st.session_state["saved_manifest_input"] = ""
        st.session_state["use_manifest_checkbox"] = False

    required_state = {
        "workflow_test_csv",
        "workflow_feature_cache",
        "workflow_run_name",
        "workflow_model_output",
        "workflow_training_output",
        "workflow_model_path",
        "workflow_prediction_output",
        "workflow_search_space",
        "workflow_parameters",
        "workflow_folds",
        "workflow_repeats",
    }
    if not required_state <= set(st.session_state):
        sample = st.session_state["workflow_dataset_choice"] == SAMPLE_DATASET
        existing_prediction_output = st.session_state.get(
            "workflow_prediction_output"
        )
        st.session_state.setdefault("workflow_test_csv", str(default_test_csv(sample)))
        st.session_state.setdefault(
            "workflow_feature_cache",
            (
                "artifacts/features/ui_sample.npz"
                if sample
                else "artifacts/features/four_sensor_v3.npz"
            ),
        )
        reset_algorithm_outputs()
        if existing_prediction_output:
            st.session_state["workflow_prediction_output"] = (
                existing_prediction_output
            )

    st.subheader("1. Dataset")
    st.radio(
        "Dataset size",
        DATASET_CHOICES,
        horizontal=True,
        key="workflow_dataset_choice",
        on_change=dataset_changed,
        help="The sample dataset is a small prepared fixture; full uses your extracted dataset.",
    )
    if st.session_state["workflow_dataset_choice"] == FULL_DATASET:
        st.text_input(
            "Full dataset root",
            key="workflow_full_dataset_root",
            on_change=full_root_changed,
            help="Must contain Training/ and Testing/ directories.",
        )
    else:
        st.text_input(
            "Sample dataset root",
            value=str(sample_root),
            disabled=True,
        )
        if not sample_root.is_dir():
            st.warning(
                "The sample has not been prepared yet. Run "
                "`python scripts/prepare_sample_dataset.py --source /path/to/small-model`."
            )

    with st.expander("Dataset details", expanded=False):
        st.text_input(
            "Test CSV",
            key="workflow_test_csv",
            help="Defines test clip order and submission paths.",
        )

    dataset_root, test_csv, sample_selected = workflow_dataset_paths(
        repository_root, default_dataset_root, st.session_state
    )
    st.session_state["dataset_root_input"] = str(dataset_root)
    st.session_state["pipeline_dataset_root"] = str(dataset_root)
    st.session_state["pipeline_test_csv"] = str(test_csv)
    dataset_ready = dataset_root.is_dir()
    if dataset_ready:
        st.success(
            f"Using {'sample' if sample_selected else 'full'} dataset: `{dataset_root}`"
        )
    else:
        st.error(f"Dataset root does not exist: {dataset_root}")

    active_run = st.session_state.get("training_pipeline_process")
    if isinstance(active_run, TrainingProcess):
        st.subheader(f"Current run: {active_run.config.run_name}")
        _render_run_monitor(active_run)
        if active_run.process.poll() is None:
            return

    started_here = (
        active_run.run_dir if isinstance(active_run, TrainingProcess) else None
    )
    for orphan in reconcile_orphaned_runs(repository_root):
        if str(orphan.get("run_dir")) == str(started_here):
            continue
        _render_orphan_run(orphan, repository_root)

    def validate_dataset() -> None:
        if not dataset_root.is_dir():
            raise ValueError(f"Dataset root does not exist: {dataset_root}")
        required_training = dataset_root / "Training" / "data" / "HAR" / "data"
        required_test = (
            dataset_root / "Testing" / "data" / "small_model_track_test"
        )
        if not required_training.is_dir():
            raise ValueError(
                f"Extracted training data was not found at {required_training}. "
                "Run scripts/prepare_training_data.sh first."
            )
        if not required_test.is_dir():
            raise ValueError(f"Extracted test data was not found at {required_test}")
        if not test_csv.is_file():
            raise ValueError(f"Test CSV does not exist: {test_csv}")

    st.divider()
    st.subheader("2. Feature extraction")
    st.caption(
        "Create one reusable feature file. Training reuses this path automatically."
    )
    with st.form("feature_extraction_form"):
        feature_cache_text = st.text_input(
            "Feature output file",
            key="workflow_feature_cache",
            help="Must be a repository-local .npz file.",
        )
        feature_columns = st.columns(3)
        feature_jobs = feature_columns[0].number_input(
            "Parallel jobs",
            min_value=1,
            max_value=64,
            value=min(8, os.cpu_count() or 1),
            key="workflow_feature_jobs",
        )
        rebuild_features = feature_columns[1].checkbox(
            "Rebuild existing cache",
            help="Replace the file even when it contains compatible features.",
        )
        allow_feature_overwrite = feature_columns[2].checkbox(
            "Allow replacement",
            help="Required when rebuilding an existing file.",
        )
        extract_submitted = st.form_submit_button(
            "Run feature extraction", type="primary", width="stretch"
        )

    if extract_submitted:
        try:
            validate_dataset()
            feature_cache = repository_output_path(
                repository_root, feature_cache_text, "Feature output"
            )
            if feature_cache.suffix.lower() != ".npz":
                raise ValueError("Feature output must use the .npz extension")
            if (
                feature_cache.exists()
                and rebuild_features
                and not allow_feature_overwrite
            ):
                raise ValueError(
                    "The feature output already exists. Enable replacement to rebuild it."
                )
            config = TrainingPipelineConfig(
                dataset_root=dataset_root,
                test_csv=test_csv,
                algorithm=str(st.session_state["workflow_algorithm"]),
                run_name=("features-sample" if sample_selected else "features-full"),
                feature_cache=feature_cache,
                artifacts_dir=repository_root / "artifacts" / "feature_extraction",
                output_path=repository_root / "outputs" / "feature_extraction.csv",
                n_jobs=int(feature_jobs),
                folds=2 if sample_selected else 5,
                cv_repeats=1,
                rebuild_features=bool(rebuild_features),
                extract_only=True,
            )
            st.session_state["training_pipeline_process"] = start_training_run(
                repository_root, config
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("3. Training")
    st.caption(
        f"Uses `{st.session_state['workflow_feature_cache']}`; if it is missing, "
        "the trainer creates it first."
    )

    def algorithm_changed() -> None:
        reset_algorithm_outputs()

    st.selectbox(
        "Algorithm",
        algorithm_names,
        format_func=lambda name: f"{algorithms[name].display_name} · {name}",
        key="workflow_algorithm",
        on_change=algorithm_changed,
    )
    algorithm_name = str(st.session_state["workflow_algorithm"])

    with st.form("model_training_form"):
        training_left, training_right = st.columns(2)
        with training_left:
            run_name_text = st.text_input("Run name", key="workflow_run_name")
            model_output_text = st.text_input(
                "Model output file",
                key="workflow_model_output",
                help="Repository-local .joblib file for the trained model.",
            )
            training_output_text = st.text_input(
                "Training prediction output",
                key="workflow_training_output",
                help="Test predictions produced by the newly trained model.",
            )
        with training_right:
            training_jobs = st.number_input(
                "Parallel feature jobs",
                min_value=1,
                max_value=64,
                value=min(8, os.cpu_count() or 1),
                key="workflow_training_jobs",
            )
            allow_training_overwrite = st.checkbox(
                "Allow replacing training outputs",
                help="Required if any model, validation, or CSV output already exists.",
            )

        with st.expander("Validation and advanced settings", expanded=False):
            run_validation = st.checkbox(
                "Run participant-grouped cross-validation", value=True
            )
            settings = st.columns(4)
            folds = settings[0].number_input(
                "Folds", min_value=2, max_value=20, key="workflow_folds"
            )
            repeats = settings[1].number_input(
                "CV repeats", min_value=1, max_value=20, key="workflow_repeats"
            )
            metric = settings[2].selectbox(
                "Selection metric",
                ("macro_f1", "accuracy", "balanced_accuracy"),
                key="workflow_metric",
            )
            random_state = settings[3].number_input(
                "Random seed",
                min_value=0,
                max_value=2_147_483_647,
                value=42,
                key="workflow_random_state",
            )
            search_space_text = st.text_area(
                "Parameter search space (JSON)",
                key="workflow_search_space",
                disabled=not run_validation,
            )
            parameters_text = st.text_area(
                "Final parameters when validation is skipped (JSON)",
                key="workflow_parameters",
                disabled=run_validation,
            )

        train_submitted = st.form_submit_button(
            "Run training", type="primary", width="stretch"
        )

    if train_submitted:
        try:
            validate_dataset()
            run_name = validate_run_name(run_name_text)
            feature_cache = repository_output_path(
                repository_root,
                str(st.session_state["workflow_feature_cache"]),
                "Feature cache",
            )
            if feature_cache.suffix.lower() != ".npz":
                raise ValueError("Feature cache must use the .npz extension")
            model_output = model_output_path(repository_root, model_output_text)
            artifacts_dir = model_output.parent
            output_path = repository_output_path(
                repository_root, training_output_text, "Training prediction output"
            )
            if output_path.suffix.lower() != ".csv":
                raise ValueError("Training prediction output must use the .csv extension")
            collisions = [
                path
                for path in (
                    model_output,
                    artifacts_dir / "validation.json",
                    artifacts_dir / "oof_predictions.npz",
                    output_path,
                )
                if path.exists()
            ]
            if collisions and not allow_training_overwrite:
                names = ", ".join(
                    str(path.relative_to(repository_root)) for path in collisions
                )
                raise ValueError(
                    f"Training would replace existing files: {names}. "
                    "Enable replacement or choose new outputs."
                )

            search_space = (
                parse_json_object(search_space_text, "Parameter search space")
                if run_validation
                else None
            )
            if search_space is not None:
                for key, values in search_space.items():
                    if not isinstance(values, list) or not values:
                        raise ValueError(
                            f"Search-space value for {key!r} must be a nonempty JSON list"
                        )
            parameters = (
                None
                if run_validation
                else parse_json_object(parameters_text, "Final parameters")
            )
            config = TrainingPipelineConfig(
                dataset_root=dataset_root,
                test_csv=test_csv,
                algorithm=algorithm_name,
                run_name=run_name,
                feature_cache=feature_cache,
                artifacts_dir=artifacts_dir,
                output_path=output_path,
                n_jobs=int(training_jobs),
                folds=int(folds),
                cv_repeats=int(repeats),
                selection_metric=str(metric),
                random_state=int(random_state),
                search_space=search_space,
                parameters=parameters,
                skip_validation=not run_validation,
                model_output_path=model_output,
            )
            st.session_state["training_pipeline_process"] = start_training_run(
                repository_root, config
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("4. Prediction")
    st.caption(
        "Run a saved model on training clips, test clips, or both. The saved CSV "
        "is used by Overview and Clip explorer."
    )
    model_artifacts = discover_model_artifacts(repository_root)
    configured_model = Path(
        str(st.session_state.get("workflow_model_path", ""))
    ).expanduser()
    if not configured_model.is_absolute():
        configured_model = repository_root / configured_model
    if model_artifacts and not configured_model.is_file():
        st.session_state["workflow_model_path"] = str(model_artifacts[0])

    with st.form("prediction_form"):
        prediction_left, prediction_right = st.columns(2)
        with prediction_left:
            model_path_text = st.text_input(
                "Model file",
                key="workflow_model_path",
                help="Defaults to the newest artifacts/*/model.joblib file.",
            )
            prediction_output_text = st.text_input(
                "Prediction output file",
                key="workflow_prediction_output",
                help="This CSV is loaded automatically by the visualization pages.",
            )
        with prediction_right:
            prediction_split = st.selectbox(
                "Split to predict",
                ("Test only", "Train only", "Both train and test"),
                key="workflow_prediction_split",
            )
            prediction_jobs = st.number_input(
                "Parallel jobs",
                min_value=1,
                max_value=64,
                value=min(8, os.cpu_count() or 1),
                key="workflow_prediction_jobs",
            )
            allow_prediction_overwrite = st.checkbox(
                "Allow replacing prediction output"
            )
        prediction_actions = st.columns(2)
        prediction_submitted = prediction_actions[0].form_submit_button(
            "Run prediction", type="primary", width="stretch"
        )
        activate_prediction = prediction_actions[1].form_submit_button(
            "Use existing CSV", width="stretch"
        )

    prediction_output_candidate = Path(prediction_output_text).expanduser()
    if not prediction_output_candidate.is_absolute():
        prediction_output_candidate = repository_root / prediction_output_candidate
    # Only adopt the file when the user asks for it. Doing this on every render
    # meant typing an existing path silently switched what the visualization
    # pages were showing, and it fought with the other writers of this key.
    if activate_prediction:
        if prediction_output_candidate.is_file():
            st.session_state["prediction_csv_input"] = str(
                prediction_output_candidate.resolve()
            )
            st.success(
                f"Using `{prediction_output_candidate.resolve()}` in the visualization pages."
            )
        else:
            st.error(f"Prediction CSV does not exist: {prediction_output_candidate}")

    if prediction_submitted:
        try:
            validate_dataset()
            model_path = Path(model_path_text).expanduser()
            if not model_path.is_absolute():
                model_path = repository_root / model_path
            model_path = model_path.resolve()
            if not model_path.is_file():
                raise ValueError(f"Model file does not exist: {model_path}")
            prediction_output = repository_output_path(
                repository_root, prediction_output_text, "Prediction output"
            )
            if prediction_output.suffix.lower() != ".csv":
                raise ValueError("Prediction output must use the .csv extension")
            # Only a test-only run produces the `path,prediction` competition
            # format. Train-only and combined files are keyed by clip_id and
            # include training rows, so writing one to a *_submission.csv name
            # would replace a gradeable file with one the grader rejects.
            reject_submission_name(prediction_split, prediction_output)
            if prediction_output.exists() and not allow_prediction_overwrite:
                raise ValueError(
                    "The prediction output already exists. Enable replacement or choose a new file."
                )

            progress_bar = st.progress(0, text="Preparing prediction…")

            def prediction_progress(done: int, total: int, message: str) -> None:
                progress_bar.progress(
                    min(100, int(100 * done / max(1, total))), text=message
                )

            if prediction_split == "Both train and test":
                split_tables = generate_all_split_predictions(
                    model_path,
                    dataset_root,
                    test_csv,
                    n_jobs=int(prediction_jobs),
                    progress_callback=prediction_progress,
                )
                table = combine_prediction_tables(split_tables)
                identifier = "clip_id"
            else:
                split = "train" if prediction_split == "Train only" else "test"
                table = generate_predictions_from_model(
                    model_path,
                    dataset_root,
                    test_csv,
                    split=split,
                    n_jobs=int(prediction_jobs),
                    progress_callback=prediction_progress,
                )
                identifier = "clip_id" if split == "train" else "path"
            save_prediction_csv(
                table,
                prediction_output,
                identifier=identifier,
            )
            progress_bar.progress(100, text="Prediction complete")
            st.session_state["prediction_csv_input"] = str(prediction_output)
            st.success(
                f"Saved {len(table.by_clip):,} predictions to `{prediction_output}`. "
                "They are now active in the visualization pages."
            )
        except Exception as exc:
            st.error(str(exc))

    active_prediction = str(st.session_state.get("prediction_csv_input", "")).strip()
    if active_prediction:
        st.info(f"Visualization prediction source: `{active_prediction}`")

        def open_clip_explorer() -> None:
            st.session_state["page_selector"] = "Clip explorer"

        st.button(
            "Visualize predictions with sensor data",
            on_click=open_clip_explorer,
            type="primary",
        )

    saved = _saved_outputs(repository_root)
    if saved:
        with st.expander("Reusable saved outputs", expanded=False):
            st.dataframe(saved, hide_index=True, width="stretch")
