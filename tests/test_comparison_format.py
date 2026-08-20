"""Tests for visualization/comparison_format.py.

The comparison format is shared by `python -m modeling.compare` and the
dashboard's Algorithm comparison page, and it is standard-library only, so
these tests always run — no dataset, no NumPy, no Streamlit.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from visualization.comparison_format import (
    COMPARISON_FIELDS,
    assign_labels,
    comparison_row,
    selected_metrics,
)


def _report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "algorithm": "logistic_regression",
        "selected_parameters": {"C": 1.0},
        "aggregate_metrics": [
            {"parameters": {"C": 0.01}, "metrics": {"accuracy": 0.4, "macro_f1": 0.35, "balanced_accuracy": 0.34}},
            {"parameters": {"C": 1.0}, "metrics": {"accuracy": 0.45, "macro_f1": 0.39, "balanced_accuracy": 0.38}},
        ],
        "training_clips": 2785,
    }
    base.update(overrides)
    return base


class SelectedMetricsTests(unittest.TestCase):
    def test_picks_the_candidate_matching_selected_parameters(self) -> None:
        self.assertEqual(selected_metrics(_report())["macro_f1"], 0.39)

    def test_unknown_selection_yields_empty(self) -> None:
        self.assertEqual(selected_metrics(_report(selected_parameters={"C": 99})), {})

    def test_skipped_validation_yields_empty(self) -> None:
        report = {"selected_parameters": {"C": 1.0}, "validation_skipped": True}
        self.assertEqual(selected_metrics(report), {})


class ComparisonRowTests(unittest.TestCase):
    def test_row_has_exactly_the_shared_fields_minus_label(self) -> None:
        row = comparison_row(Path("/tmp/logreg/validation.json"), _report())
        self.assertEqual(set(row) | {"label"}, set(COMPARISON_FIELDS))

    def test_metrics_come_from_the_selected_candidate(self) -> None:
        row = comparison_row(Path("/tmp/logreg/validation.json"), _report())
        self.assertEqual(row["accuracy"], 0.45)
        self.assertEqual(row["macro_f1"], 0.39)
        self.assertEqual(row["balanced_accuracy"], 0.38)
        self.assertEqual(row["parameters"], '{"C": 1.0}')

    def test_missing_algorithm_falls_back_to_artifact_dir(self) -> None:
        row = comparison_row(Path("/tmp/rbf_svm/validation.json"), {"selected_parameters": {}})
        self.assertEqual(row["algorithm"], "rbf_svm")
        self.assertEqual(row["artifact_name"], "rbf_svm")

    def test_skipped_validation_leaves_metrics_none(self) -> None:
        row = comparison_row(Path("/tmp/x/validation.json"), {"selected_parameters": {}, "validation_skipped": True})
        self.assertIsNone(row["accuracy"])
        self.assertIsNone(row["macro_f1"])


class AssignLabelsTests(unittest.TestCase):
    def test_unique_algorithms_keep_bare_names(self) -> None:
        rows = [
            {"algorithm": "logistic_regression", "artifact_name": "logreg"},
            {"algorithm": "rbf_svm", "artifact_name": "svm"},
        ]
        assign_labels(rows)
        self.assertEqual([r["label"] for r in rows], ["logistic_regression", "rbf_svm"])

    def test_duplicate_algorithms_are_disambiguated(self) -> None:
        """Two artifact dirs from the same algorithm must not collide."""
        rows = [
            {"algorithm": "logistic_regression", "artifact_name": "logreg"},
            {"algorithm": "logistic_regression", "artifact_name": "logreg_tuned"},
            {"algorithm": "rbf_svm", "artifact_name": "svm"},
        ]
        assign_labels(rows)
        self.assertEqual(
            [r["label"] for r in rows],
            ["logistic_regression (logreg)", "logistic_regression (logreg_tuned)", "rbf_svm"],
        )

    def test_identical_algorithm_and_artifact_still_unique(self) -> None:
        rows = [{"algorithm": "a", "artifact_name": "dir"}, {"algorithm": "a", "artifact_name": "dir"}]
        assign_labels(rows)
        self.assertEqual(len({r["label"] for r in rows}), 2)

    def test_labels_do_not_depend_on_row_order_of_other_algorithms(self) -> None:
        forward = [{"algorithm": "a", "artifact_name": "x"}, {"algorithm": "b", "artifact_name": "y"}]
        backward = [{"algorithm": "b", "artifact_name": "y"}, {"algorithm": "a", "artifact_name": "x"}]
        assign_labels(forward)
        assign_labels(backward)
        self.assertEqual({r["algorithm"]: r["label"] for r in forward}, {r["algorithm"]: r["label"] for r in backward})


class CliParityTests(unittest.TestCase):
    """`modeling.compare` must write exactly the shared fields."""

    def test_cli_uses_the_shared_fields(self) -> None:
        try:
            from modeling import compare
        except ImportError as exc:  # modeling/__init__ needs numpy + scikit-learn
            self.skipTest(f"modeling package unavailable ({exc})")
        self.assertIs(compare.COMPARISON_FIELDS, COMPARISON_FIELDS)

    def test_cli_row_matches_shared_row(self) -> None:
        try:
            from modeling import compare
        except ImportError as exc:
            self.skipTest(f"modeling package unavailable ({exc})")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logreg" / "validation.json"
            path.parent.mkdir()
            report = _report()
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(compare._row(path), comparison_row(path, report))


if __name__ == "__main__":
    unittest.main()
