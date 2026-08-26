"""Tests for visualization/algorithm_comparison.py.

Covers discovery, leaderboard columns, CSV generation, and figure helpers
without needing the dataset. Heavy deps (numpy, pandas, plotly) are optional;
the suite skips gracefully if they are absent (same pattern as test_modeling).
The dataset-free, dependency-free half of the format lives in
`tests/test_comparison_format.py`, which always runs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from visualization.comparison_format import COMPARISON_FIELDS, selected_metrics

try:
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import plotly  # noqa: F401

    from visualization.algorithm_comparison import (
        METRIC_CHOICES,
        REPOSITORY_ROOT,
        _class_labels,
        _confusion_figure,
        _delta_figure,
        _fold_metric_series,
        _leaderboard_csv,
        cached_validation_reports,
        report_cache_keys,
        _row,
        discover_validation_reports,
        display_columns,
    )

    HAS_DEPS = True
except ImportError as exc:  # pragma: no cover
    HAS_DEPS = False
    _IMPORT_ERROR = exc


if not HAS_DEPS:

    class _Skipped(unittest.TestCase):
        def test_skipped(self) -> None:
            raise unittest.SkipTest(f"algorithm comparison tests need visualization deps ({_IMPORT_ERROR})")

else:

    class DiscoveryTests(unittest.TestCase):
        def test_discovers_per_algorithm_reports(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "logreg").mkdir()
                (root / "rbf_svm").mkdir()
                (root / "logreg" / "validation.json").write_text(json.dumps({"algorithm": "logreg"}))
                (root / "rbf_svm" / "validation.json").write_text(json.dumps({"algorithm": "rbf"}))
                paths = discover_validation_reports(root)
                self.assertEqual(len(paths), 2)
                self.assertTrue(all(p.name == "validation.json" for p in paths))
                self.assertEqual(sorted(p.parent.name for p in paths), ["logreg", "rbf_svm"])

        def test_empty_when_no_reports(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(discover_validation_reports(Path(tmp)), [])

        def test_flat_layout_fallback(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "validation.json").write_text(json.dumps({"algorithm": "flat"}))
                paths = discover_validation_reports(root)
                self.assertEqual(len(paths), 1)
                self.assertEqual(paths[0].name, "validation.json")

    class ParsingTests(unittest.TestCase):
        def _report(self, **overrides) -> dict:
            base = {
                "algorithm": "logistic_regression",
                "algorithm_display_name": "LogReg",
                "selected_parameters": {"C": 1.0},
                "aggregate_metrics": [
                    {"parameters": {"C": 0.01}, "metrics": {"accuracy": 0.4, "macro_f1": 0.35, "balanced_accuracy": 0.34}},
                    {"parameters": {"C": 1.0}, "metrics": {"accuracy": 0.45, "macro_f1": 0.39, "balanced_accuracy": 0.38}},
                ],
                "confusion_matrix": [[2, 0], [0, 2]],
                "training_clips": 2785,
                "test_clips": 405,
                "library_versions": {"scikit-learn": "1.5.0", "numpy": "2.0.0"},
            }
            base.update(overrides)
            return base

        def test_row_carries_every_shared_field(self) -> None:
            """The dashboard row must stay a superset of the CSV/CLI columns."""
            row = _row(Path("/tmp/logreg/validation.json"), self._report())
            self.assertEqual(set(COMPARISON_FIELDS) - {"label"} - set(row), set())
            self.assertEqual(row["macro_f1"], 0.39)

        def test_row_skipped_validation_has_no_metrics(self) -> None:
            report = {"algorithm": "x", "selected_parameters": {"C": 1.0}, "validation_skipped": True}
            row = _row(Path("/tmp/a/validation.json"), report)
            self.assertIsNone(row["macro_f1"])
            self.assertTrue(row["validation_skipped"])

        def test_row_uses_display_name_fallback(self) -> None:
            report = {"algorithm": "my_algo", "selected_parameters": {}}
            row = _row(Path("/tmp/my_algo/validation.json"), report)
            self.assertEqual(row["display_name"], "my_algo")

        def test_row_falls_back_to_artifact_dir_without_algorithm_key(self) -> None:
            row = _row(Path("/tmp/rbf_svm/validation.json"), {"selected_parameters": {}})
            self.assertEqual(row["algorithm"], "rbf_svm")
            self.assertEqual(row["artifact_name"], "rbf_svm")
            self.assertEqual(row["display_name"], "rbf_svm")

    class LeaderboardTests(unittest.TestCase):
        def test_display_columns_are_deduplicated(self) -> None:
            """Sort metric is already in the fixed list; it must not repeat."""
            row = {
                "label": "a",
                "algorithm": "a",
                "display_name": "A",
                "artifact_name": "dir_a",
                "parameters": "{}",
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "balanced_accuracy": 0.45,
                "training_clips": 100,
                "report": "r1",
            }
            for sort_metric in METRIC_CHOICES:
                columns = display_columns(sort_metric)
                self.assertEqual(len(columns), len(set(columns)), sort_metric)
                self.assertEqual(set(columns), set(row), sort_metric)
                # Selecting them must not raise "Duplicate column names found"
                pd.DataFrame([row])[columns]

        def test_display_columns_lead_with_sort_metric(self) -> None:
            columns = display_columns("accuracy")
            self.assertLess(columns.index("accuracy"), columns.index("macro_f1"))

        def test_leaderboard_csv_matches_the_shared_fields(self) -> None:
            """The download CSV header is the one `modeling.compare` writes."""
            rows = [
                {"label": "a", "algorithm": "a", "artifact_name": "d1", "parameters": "{}", "accuracy": 0.5, "macro_f1": 0.4, "balanced_accuracy": 0.45, "training_clips": 10, "report": "r1"},
                {"label": "b", "algorithm": "b", "artifact_name": "d2", "parameters": "{}", "accuracy": 0.6, "macro_f1": 0.5, "balanced_accuracy": 0.55, "training_clips": 20, "report": "r2"},
            ]
            text = _leaderboard_csv(rows)
            self.assertEqual(text.splitlines()[0], ",".join(COMPARISON_FIELDS))
            self.assertEqual(len(text.splitlines()), 3)

        def test_leaderboard_csv_tolerates_dashboard_only_keys(self) -> None:
            """Rows carry extras like `display_name`; DictWriter must not raise."""
            row = {field: "x" for field in COMPARISON_FIELDS}
            row["display_name"] = "ignored"
            row["parameters_dict"] = {"C": 1.0}
            self.assertEqual(_leaderboard_csv([row]).splitlines()[0], ",".join(COMPARISON_FIELDS))

        def test_real_scores_reference(self) -> None:
            """Guard the reference real scores we now ship in artifacts/logreg/validation.json."""
            path = REPOSITORY_ROOT / "artifacts" / "logreg" / "validation.json"
            if not path.is_file():
                self.skipTest("No real validation.json yet (run modeling.train first)")
            report = json.loads(path.read_text())
            self.assertEqual(report["selected_parameters"], {"C": 0.1})
            # ``metrics`` is the mean over repeats, so each repeat count has its
            # own reference. An unrecorded count must fail rather than fall
            # through to whichever branch happens to be left.
            reference = {
                1: {"macro_f1": 0.4765, "accuracy": 0.5382},
                3: {"macro_f1": 0.4751, "accuracy": 0.5370},
            }
            repeats = report.get("cv_repeats", 1)
            self.assertIn(repeats, reference, f"No reference scores for cv_repeats={repeats}")
            metrics = selected_metrics(report)
            for name, expected in reference[repeats].items():
                self.assertAlmostEqual(metrics[name], expected, places=3)

    class FigureTests(unittest.TestCase):
        def test_confusion_figure_shapes(self) -> None:
            cm = [[2, 1], [0, 2]]
            fig = _confusion_figure(cm, ["0", "1"], "test")
            self.assertEqual(fig.data[0].z.shape, (2, 2))
            fig_norm = _confusion_figure(cm, ["0", "1"], "test", row_normalize=True)
            # Row-normalized: first row sums to 1
            import numpy as np

            arr = np.asarray(fig_norm.data[0].z)
            self.assertAlmostEqual(float(arr[0].sum()), 1.0, places=5)

        def test_delta_figure_is_centered_at_zero(self) -> None:
            a = [[2, 0], [0, 2]]
            b = [[1, 1], [1, 1]]
            fig = _delta_figure(a, b, ["0", "1"], "delta")
            self.assertEqual(fig.data[0].z.shape, (2, 2))
            # Diverging scale — plotly expands "RdBu" to a tuple, so check z is centered
            self.assertAlmostEqual(float(fig.data[0].zmin), -float(fig.data[0].zmax), places=5)
            self.assertEqual(fig.data[0].zmid, 0)

        def test_class_labels_fallback(self) -> None:
            report = {"confusion_matrix": [[0] * 3] * 3}
            labels = _class_labels(report)
            self.assertEqual(len(labels), 3)

        def test_fold_series(self) -> None:
            report = {
                "selected_parameters": {"C": 1.0},
                "folds": [
                    {"fold": 1, "candidates": [{"parameters": {"C": 1.0}, "metrics": {"macro_f1": 0.5}}]},
                    {"fold": 2, "candidates": [{"parameters": {"C": 1.0}, "metrics": {"macro_f1": 0.6}}]},
                ],
            }
            df = _fold_metric_series(report, "macro_f1")
            self.assertEqual(list(df["fold"]), [1, 2])
            self.assertEqual(list(df["macro_f1"]), [0.5, 0.6])

    class ReportCacheKeyTests(unittest.TestCase):
        """Retraining rewrites validation.json at the same path."""

        def test_key_changes_when_a_report_is_rewritten(self) -> None:
            import os

            with tempfile.TemporaryDirectory() as temporary:
                report = Path(temporary) / "validation.json"
                report.write_text('{"macro_f1": 0.4}', encoding="utf-8")
                os.utime(report, ns=(1_000_000_000, 1_000_000_000))
                before = report_cache_keys([report])

                report.write_text('{"macro_f1": 0.9}', encoding="utf-8")
                os.utime(report, ns=(2_000_000_000, 2_000_000_000))
                after = report_cache_keys([report])

            # Same path both times; only the stat pair separates the two runs,
            # so a path-only key would serve the stale 0.4 report.
            self.assertEqual(before[0][0], after[0][0])
            self.assertNotEqual(before, after)

        def test_unreadable_reports_still_produce_a_key(self) -> None:
            with tempfile.TemporaryDirectory() as temporary:
                missing = Path(temporary) / "gone" / "validation.json"
                self.assertEqual(report_cache_keys([missing]), ((str(missing), 0, 0),))

        def test_cached_reports_are_keyed_by_the_stat_tuple(self) -> None:
            with tempfile.TemporaryDirectory() as temporary:
                report = Path(temporary) / "validation.json"
                report.write_text('{"algorithm": "logreg"}', encoding="utf-8")
                reports = cached_validation_reports(report_cache_keys([report]))

            self.assertEqual(reports[0]["algorithm"], "logreg")
            self.assertEqual(reports[0]["_path"], str(report))


if __name__ == "__main__":
    unittest.main()
