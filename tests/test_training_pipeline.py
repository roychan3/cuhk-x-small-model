from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_sample_dataset import evenly_spaced
from visualization.training_pipeline import (
    TrainingPipelineConfig,
    build_training_command,
    next_available_run_name,
    overall_progress,
    parse_json_object,
    reconcile_orphaned_runs,
    repository_output_path,
    stage_states,
    validate_run_name,
)
from visualization.progress import COMPLETE, ERROR, RUNNING, mark_error, write_progress


class TrainingPipelineConfigurationTests(unittest.TestCase):
    def _config(self, root: Path, **overrides: object) -> TrainingPipelineConfig:
        values: dict[str, object] = {
            "dataset_root": root / "dataset",
            "test_csv": root / "Testing" / "test.csv",
            "algorithm": "logistic_regression",
            "run_name": "logreg-ui",
            "feature_cache": root / "artifacts" / "features" / "features.npz",
            "artifacts_dir": root / "artifacts" / "logreg-ui",
            "output_path": root / "outputs" / "logreg-ui_submission.csv",
            "n_jobs": 3,
            "folds": 4,
            "cv_repeats": 2,
            "selection_metric": "balanced_accuracy",
            "random_state": 9,
            "search_space": {"C": [0.1, 1.0]},
        }
        values.update(overrides)
        return TrainingPipelineConfig(**values)  # type: ignore[arg-type]

    def test_full_pipeline_command_uses_cli_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = root / "progress.json"
            command = build_training_command(self._config(root), progress)

        self.assertEqual(command[:4], [command[0], "-u", "-m", "modeling.train"])
        self.assertEqual(
            command[command.index("--algorithm") + 1], "logistic_regression"
        )
        self.assertEqual(command[command.index("--folds") + 1], "4")
        self.assertEqual(command[command.index("--cv-repeats") + 1], "2")
        self.assertEqual(
            json.loads(command[command.index("--search-space") + 1]), {"C": [0.1, 1.0]}
        )
        self.assertNotIn("--skip-validation", command)
        self.assertNotIn("--extract-only", command)

    def test_extract_only_and_fixed_parameters_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extract = build_training_command(
                self._config(root, extract_only=True, search_space=None),
                root / "extract.json",
            )
            fixed = build_training_command(
                self._config(
                    root,
                    skip_validation=True,
                    search_space=None,
                    parameters={"C": 0.25},
                ),
                root / "fixed.json",
            )

        self.assertIn("--extract-only", extract)
        self.assertIn("--skip-validation", fixed)
        self.assertEqual(
            json.loads(fixed[fixed.index("--parameters") + 1]), {"C": 0.25}
        )

    def test_output_paths_cannot_escape_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(
                repository_output_path(root, "artifacts/features/cache.npz", "cache"),
                root / "artifacts" / "features" / "cache.npz",
            )
            with self.assertRaisesRegex(ValueError, "inside the repository"):
                repository_output_path(root, "../outside.npz", "cache")

    def test_run_names_are_safe_and_increment_around_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifacts" / "logreg").mkdir(parents=True)
            self.assertEqual(next_available_run_name(root, "logreg"), "logreg-2")
        self.assertEqual(validate_run_name("late_fusion.2"), "late_fusion.2")
        with self.assertRaises(ValueError):
            validate_run_name("../escape")

    def test_json_fields_require_objects(self) -> None:
        self.assertIsNone(parse_json_object("", "settings"))
        self.assertEqual(parse_json_object('{"C": [1]}', "settings"), {"C": [1]})
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_json_object("[1, 2]", "settings")

    def test_shared_progress_writer_creates_timestamped_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "progress.json"
            write_progress(path, status=RUNNING, stage="features", current=3, total=10)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], RUNNING)
        self.assertEqual(payload["stage"], "features")
        self.assertEqual(payload["current"], 3)
        self.assertEqual(payload["total"], 10)
        self.assertIn("updated_at", payload)

    def test_mark_error_keeps_the_stage_the_job_died_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.json"
            write_progress(path, status=RUNNING, stage="validation", current=2, total=5)
            mark_error(path, "RuntimeError: boom")
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], ERROR)
        self.assertEqual(payload["message"], "RuntimeError: boom")
        # Reporting the failure must not lose where it happened.
        self.assertEqual(payload["stage"], "validation")
        self.assertEqual(payload["current"], 2)
        self.assertEqual(payload["total"], 5)

    def test_mark_error_tolerates_a_missing_or_corrupt_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "absent.json"
            mark_error(missing, "died before the first write")
            self.assertEqual(
                json.loads(missing.read_text(encoding="utf-8"))["status"], ERROR
            )

            corrupt = Path(temporary) / "corrupt.json"
            corrupt.write_text("{not json", encoding="utf-8")
            mark_error(corrupt, "still recorded")
            self.assertEqual(
                json.loads(corrupt.read_text(encoding="utf-8"))["status"], ERROR
            )

    def test_sample_builder_keeps_evenly_spaced_files(self) -> None:
        paths = [Path(f"frame-{index:02d}.png") for index in range(10)]
        self.assertEqual(
            evenly_spaced(paths, 4),
            [paths[0], paths[3], paths[6], paths[9]],
        )


class TrainingPipelineProgressTests(unittest.TestCase):
    def test_feature_and_validation_progress_are_scaled(self) -> None:
        self.assertAlmostEqual(
            overall_progress({"stage": "features", "current": 50, "total": 100}, None),
            0.28,
        )
        self.assertAlmostEqual(
            overall_progress({"stage": "validation", "current": 5, "total": 10}, None),
            0.65,
        )
        self.assertEqual(overall_progress({"stage": "complete"}, 0), 1.0)

    def test_stage_states_mark_skipped_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = TrainingPipelineConfigurationTests()._config(
                Path(temporary), extract_only=True
            )
        states = dict(stage_states({"stage": "complete"}, config, 0))
        self.assertEqual(states["Discover dataset"], "completed")
        self.assertEqual(states["Extract or load features"], "completed")
        self.assertEqual(states["Validate parameters"], "skipped")
        self.assertEqual(states["Fit final model"], "skipped")

    def test_reported_error_marks_the_failing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = TrainingPipelineConfigurationTests()._config(Path(temporary))
        # The process is still being reaped, so only the file reports the failure.
        states = dict(
            stage_states({"status": ERROR, "stage": "validation"}, config, None)
        )
        self.assertEqual(states["Discover dataset"], "completed")
        self.assertEqual(states["Validate parameters"], "failed")
        self.assertEqual(states["Fit final model"], "pending")

    def test_completion_is_read_from_status_not_the_stage_name(self) -> None:
        self.assertEqual(overall_progress({"status": COMPLETE}, None), 1.0)
        # A stage called "complete" without the status is not yet finished.
        self.assertLess(overall_progress({"stage": "saving"}, None), 1.0)


if __name__ == "__main__":
    unittest.main()


class OrphanedRunReconciliationTests(unittest.TestCase):
    """A run outlives the browser session that started it."""

    def _run_record(self, root: Path, name: str, pid: int) -> Path:
        run_dir = root / "artifacts" / "training_runs" / name
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps({"status": "running", "pid": pid, "config": {"run_name": name}}),
            encoding="utf-8",
        )
        return run_dir

    def test_dead_runs_are_marked_interrupted_rather_than_left_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._run_record(root, "logreg", pid=1)
            with patch(
                "visualization.training_pipeline.process_is_alive", return_value=False
            ):
                alive = reconcile_orphaned_runs(root)

            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

        self.assertEqual(alive, [])
        self.assertEqual(record["status"], "interrupted")
        self.assertIn("finished_at", record)

    def test_live_runs_are_returned_so_a_new_session_can_watch_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._run_record(root, "late-fusion", pid=4242)
            with patch(
                "visualization.training_pipeline.process_is_alive", return_value=True
            ):
                alive = reconcile_orphaned_runs(root)

            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

        self.assertEqual(len(alive), 1)
        self.assertEqual(alive[0]["run_dir"], str(run_dir))
        # A live run must not be rewritten as finished.
        self.assertEqual(record["status"], "running")

    def test_finished_records_and_unreadable_files_are_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "artifacts" / "training_runs"
            (runs / "done").mkdir(parents=True)
            (runs / "done" / "run.json").write_text(
                json.dumps({"status": "completed", "pid": 1}), encoding="utf-8"
            )
            (runs / "broken").mkdir(parents=True)
            (runs / "broken" / "run.json").write_text("{oops", encoding="utf-8")

            self.assertEqual(reconcile_orphaned_runs(root), [])
            self.assertEqual(
                json.loads((runs / "done" / "run.json").read_text())["status"],
                "completed",
            )

    def test_missing_runs_directory_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(reconcile_orphaned_runs(Path(temporary)), [])
