from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from visualization.predictions import (
    clip_id_from_submission_path,
    discover_prediction_csvs,
    load_action_mapping,
    parse_prediction_csv,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PredictionParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = {0: "0_Wash_face", 34: "34_Sit_down"}

    def test_normalizes_submission_paths_and_skips_blank_predictions(self) -> None:
        table = parse_prediction_csv(
            "path,prediction\n"
            "small_model_track_test/SM_test_0001/,34\n"
            "small_model_track_test\\SM_test_0002\\,0\n"
            "small_model_track_test/SM_test_0003/,\n",
            self.actions,
        )

        self.assertEqual(table.rows_read, 3)
        self.assertEqual(table.blank_predictions, 1)
        self.assertEqual(set(table.by_clip), {"SM_test_0001", "SM_test_0002"})
        self.assertEqual(table.by_clip["SM_test_0001"].action_name, "34_Sit_down")
        self.assertEqual(clip_id_from_submission_path("SM_test_0042/"), "SM_test_0042")

    def test_rejects_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "path and prediction"):
            parse_prediction_csv("clip,label\na,0\n", self.actions)

    def test_rejects_unknown_or_non_integer_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown action_id 39"):
            parse_prediction_csv("path,prediction\nSM_test_0001,39\n", self.actions)
        with self.assertRaisesRegex(ValueError, "integer action_id"):
            parse_prediction_csv("path,prediction\nSM_test_0001,34.0\n", self.actions)

    def test_rejects_duplicate_clip_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate prediction for SM_test_0001"):
            parse_prediction_csv(
                "path,prediction\nSM_test_0001,34\nfolder/SM_test_0001/,0\n",
                self.actions,
            )

    def test_loads_action_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "classes.csv"
            path.write_text(
                "action_id,action_name\n0,0_Wash_face\n34,34_Sit_down\n",
                encoding="utf-8",
            )
            self.assertEqual(load_action_mapping(path), self.actions)

    def test_every_repository_action_round_trips_through_predictions(self) -> None:
        mapping = load_action_mapping(REPOSITORY_ROOT / "Training" / "class_mapping.csv")
        self.assertEqual(set(mapping), set(range(40)))

        rows = ["path,prediction"]
        rows.extend(
            f"small_model_track_test/SM_test_{action_id:04d}/,{action_id}"
            for action_id in sorted(mapping)
        )
        table = parse_prediction_csv("\n".join(rows), mapping)

        self.assertEqual(len(table.by_clip), 40)
        for action_id, action_name in mapping.items():
            prediction = table.by_clip[f"SM_test_{action_id:04d}"]
            self.assertEqual(prediction.action_id, action_id)
            self.assertEqual(prediction.action_name, action_name)

    def test_discovers_newest_generated_csv_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary) / "outputs"
            outputs.mkdir()
            older = outputs / "older_submission.csv"
            newer = outputs / "newer_submission.csv"
            unrelated = outputs / "metrics.csv"
            older.write_text("path,prediction\n", encoding="utf-8")
            newer.write_text("path,prediction\n", encoding="utf-8")
            unrelated.write_text("metric,value\naccuracy,0.5\n", encoding="utf-8")
            os.utime(older, ns=(1_000_000_000, 1_000_000_000))
            os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

            self.assertEqual(discover_prediction_csvs(temporary), [newer, older])


if __name__ == "__main__":
    unittest.main()
