from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from visualization.manual_labels import (
    archive_manual_label_file,
    initial_label_clip,
    load_manual_label_rows,
    manual_label_path,
    next_unlabeled_clip,
    write_manual_label_rows,
)


class ManualLabelTests(unittest.TestCase):
    def test_output_path_uses_manual_label_suffix(self) -> None:
        self.assertEqual(
            manual_label_path("/dataset/Testing/test.csv"),
            Path("/dataset/Testing/test_manual_label.csv"),
        )

    def test_new_label_table_does_not_copy_model_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.csv"
            source.write_text(
                "path,prediction\nsmall_model_track_test/SM_test_0001/,34\n",
                encoding="utf-8",
            )

            fields, rows = load_manual_label_rows(source, valid_action_ids=range(40))

            self.assertEqual(fields, ["path", "prediction"])
            self.assertEqual(rows[0]["prediction"], "")
            self.assertFalse(manual_label_path(source).exists())

    def test_written_labels_resume_from_manual_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.csv"
            source.write_text(
                "path,prediction\n"
                "small_model_track_test/SM_test_0001/,\n"
                "small_model_track_test/SM_test_0002/,\n",
                encoding="utf-8",
            )
            output = manual_label_path(source)
            fields, rows = load_manual_label_rows(source)
            rows[0]["prediction"] = "34"

            write_manual_label_rows(output, fields, rows)
            resumed_fields, resumed_rows = load_manual_label_rows(
                source, valid_action_ids={0, 34}
            )

            self.assertEqual(resumed_fields, fields)
            self.assertEqual([row["prediction"] for row in resumed_rows], ["34", ""])
            with output.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_rejects_manual_file_for_different_test_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.csv"
            output = manual_label_path(source)
            source.write_text("path,prediction\na/,\nb/,\n", encoding="utf-8")
            output.write_text("path,prediction\nb/,1\na/,2\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_manual_label_rows(source, valid_action_ids=range(40))

    def test_rejects_invalid_saved_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.csv"
            output = manual_label_path(source)
            source.write_text("path,prediction\na/,\n", encoding="utf-8")
            output.write_text("path,prediction\na/,99\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unknown manual action_id 99"):
                load_manual_label_rows(source, valid_action_ids=range(40))

    def test_refuses_to_treat_the_test_csv_as_its_own_label_file(self) -> None:
        # Aliasing would keep the model predictions already in the source and
        # then overwrite the test index on the next save.
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.csv"
            source.write_text("path,prediction\na/,7\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not be written over"):
                load_manual_label_rows(source, source, valid_action_ids=range(40))

    def test_rejects_rows_with_more_columns_than_the_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.csv"
            source.write_text("path,prediction\na/,1,extra\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "more columns"):
                load_manual_label_rows(source, valid_action_ids=range(40))

    def test_rejects_rows_with_fewer_columns_than_the_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.csv"
            source.write_text("path,prediction\na/\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "fewer columns"):
                load_manual_label_rows(source, valid_action_ids=range(40))


class ManualLabelWriteBackTests(unittest.TestCase):
    """The writer must only emit tables the loader will accept."""

    def test_write_rejects_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "test_manual_label.csv"
            rows = [{"path": "a/", "prediction": "1"}, {"path": "a/", "prediction": "2"}]

            with self.assertRaisesRegex(ValueError, "duplicate paths"):
                write_manual_label_rows(output, ["path", "prediction"], rows)
            self.assertFalse(output.exists())

    def test_write_rejects_unknown_action_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "test_manual_label.csv"
            rows = [{"path": "a/", "prediction": "99"}]

            with self.assertRaisesRegex(ValueError, "Unknown manual action_id 99"):
                write_manual_label_rows(
                    output, ["path", "prediction"], rows, valid_action_ids=range(40)
                )
            self.assertFalse(output.exists())

    def test_write_rejects_non_integer_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "test_manual_label.csv"
            rows = [{"path": "a/", "prediction": "walking"}]

            with self.assertRaisesRegex(ValueError, "must be an integer"):
                write_manual_label_rows(output, ["path", "prediction"], rows)
            self.assertFalse(output.exists())

    def test_archive_moves_an_unreadable_file_aside(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.csv"
            source.write_text("path,prediction\na/,\nb/,\n", encoding="utf-8")
            output = manual_label_path(source)
            output.write_text("path,prediction\nb/,1\na/,2\n", encoding="utf-8")

            backup = archive_manual_label_file(output)

            self.assertFalse(output.exists())
            self.assertTrue(backup.exists())
            self.assertIn(".invalid-", backup.name)
            self.assertEqual(backup.suffix, ".csv")
            # With the bad file out of the way the page can start over.
            _fields, rows = load_manual_label_rows(source, valid_action_ids=range(40))
            self.assertEqual([row["prediction"] for row in rows], ["", ""])


class ClipNavigationTests(unittest.TestCase):
    CLIPS = ("a", "b", "c", "d")

    @staticmethod
    def _rows(*predictions: str) -> list[dict[str, str]]:
        return [{"path": f"{p}/", "prediction": value} for p, value in
                zip("abcd", predictions, strict=True)]

    def test_opens_on_the_first_unlabeled_clip(self) -> None:
        rows = self._rows("1", "", "3", "")
        self.assertEqual(initial_label_clip(self.CLIPS, rows), "b")

    def test_opens_on_the_first_clip_when_everything_is_labeled(self) -> None:
        rows = self._rows("1", "2", "3", "4")
        self.assertEqual(initial_label_clip(self.CLIPS, rows), "a")

    def test_advances_to_the_next_unlabeled_clip(self) -> None:
        rows = self._rows("1", "2", "", "")
        self.assertEqual(next_unlabeled_clip(self.CLIPS, rows, 0), "c")

    def test_advance_wraps_past_the_end_to_an_earlier_gap(self) -> None:
        rows = self._rows("", "2", "3", "4")
        self.assertEqual(next_unlabeled_clip(self.CLIPS, rows, 3), "a")

    def test_advance_skips_the_clip_just_saved(self) -> None:
        # Only the current clip is unlabeled, so wrapping must not return it.
        rows = self._rows("1", "", "3", "4")
        self.assertEqual(next_unlabeled_clip(self.CLIPS, rows, 1), "c")

    def test_advance_falls_back_to_the_neighbour_when_all_labeled(self) -> None:
        rows = self._rows("1", "2", "3", "4")
        self.assertEqual(next_unlabeled_clip(self.CLIPS, rows, 3), "a")


if __name__ == "__main__":
    unittest.main()
