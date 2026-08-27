"""Tests for in-UI model inference and prediction attachment.

``visualization.predictions`` is importable with the standard library alone, so
artifact discovery and CSV round-tripping always run. The heavier cases need
``numpy``/``pandas``/``plotly``/``streamlit`` (for ``visualization.app``) or the
modeling stack; those skip gracefully, matching test_algorithm_comparison and
test_modeling.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from visualization.predictions import (
    ClipPrediction,
    PredictionTable,
    _prediction_table_from_arrays,
    combine_prediction_tables,
    discover_model_artifacts,
    generate_all_split_predictions,
    generate_predictions_from_model,
    load_action_mapping,
    prediction_csv_text,
    save_prediction_csv,
)

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError as exc:  # pragma: no cover
    HAS_NUMPY = False
    _NUMPY_ERROR = exc

try:
    import pandas as pd

    from visualization.app import (
        SKELETON_EDGES,
        SKELETON_JOINT_NAMES,
        IMAGE_ENCODING_STEPS,
        _clip_player_html,
        _clip_player_model,
        _image_data_uri,
        _image_encoding,
        _skeleton_edge_chain,
        _selected_paths,
        _skeleton_people,
        _stabilize_skeleton_points,
        _timeline_selections,
        add_predictions,
        correctness_flag,
        prediction_file_options,
        skeleton_calibration,
    )

    HAS_APP_DEPS = True
except ImportError as exc:  # pragma: no cover
    HAS_APP_DEPS = False
    _APP_ERROR = exc

try:
    import modeling.features  # noqa: F401

    HAS_MODELING = True
except ImportError as exc:  # pragma: no cover
    HAS_MODELING = False
    _MODELING_ERROR = exc


needs_numpy = unittest.skipUnless(HAS_NUMPY, "needs numpy")
needs_app = unittest.skipUnless(HAS_APP_DEPS, "needs visualization/requirements.txt")
needs_modeling = unittest.skipUnless(HAS_MODELING, "needs modeling/requirements.txt")


@needs_app
class SkeletonPlaybackTests(unittest.TestCase):
    def test_clip_ranges_keep_scale_fixed_between_frames(self) -> None:
        first_points = [[float(index), 0.0, float(index * 2)] for index in range(17)]
        second_points = [[float(index * 2), 0.5, float(index)] for index in range(17)]
        payloads = [
            json.dumps([{"keypoints": points, "keypoint_scores": [1.0] * 17}]).encode()
            for points in (first_points, second_points)
        ]

        axis_ranges, bone_lengths = skeleton_calibration(payloads)
        self.assertIsNotNone(axis_ranges)
        self.assertIsNotNone(bone_lengths)
        assert axis_ranges is not None
        spans = [high - low for low, high in axis_ranges]
        self.assertAlmostEqual(spans[0], spans[1])
        self.assertAlmostEqual(spans[1], spans[2])

    def test_clip_player_preloads_frames_for_in_place_browser_playback(self) -> None:
        selections = _timeline_selections(
            {
                "Depth_Color": ["depth-0.png", "depth-1.png"],
                "IR": ["ir.png"],
                "Skeleton": [],
            },
            2,
        )
        self.assertEqual(
            _selected_paths(selections, ("Depth_Color", "IR", "Thermal")),
            ("depth-0.png", "ir.png", "depth-1.png"),
        )
        # A modality shorter than the timeline repeats rather than running out.
        self.assertEqual([s["IR"] for s in selections], ["ir.png", "ir.png"])
        self.assertEqual(_selected_paths(selections, ("Skeleton",)), ())
        model = _clip_player_model(
            selections,
            {"depth-0.png": "data:image/jpeg;base64,AAA", "depth-1.png": "data:image/jpeg;base64,BBB"},
            {},
            b"frame,x,y,z,v,snr\n7,1,2,3,-0.5,60\n",
            None,
            None,
            "test:clip-1",
            {"0.5": 200, "1": 100, "2": 50},
        )

        self.assertEqual(len(model["frames"]), 2)
        # ir.png had no asset, so the frame still names it but the player skips it.
        self.assertEqual(sorted(model["imageAssets"]), ["depth-0.png", "depth-1.png"])
        self.assertEqual(model["frames"][0]["images"]["IR"], "ir.png")
        self.assertEqual(model["radarFrames"][0]["frame"], 7)
        self.assertEqual(model["intervalsMs"]["2"], 50)

    def test_repeated_skeleton_path_is_parsed_once_per_clip(self) -> None:
        """A short pose stream maps many timeline frames onto one file."""

        points = [[float(index), 0.0, float(index * 2)] for index in range(17)]
        payload = json.dumps([{"keypoints": points, "keypoint_scores": [1.0] * 17}]).encode()
        selections = [{"Skeleton": "pose.json"} for _ in range(4)]

        with patch("visualization.app._skeleton_people", wraps=_skeleton_people) as parsed:
            model = _clip_player_model(
                selections, {}, {"pose.json": payload}, None, None, None, "test:clip-1", {}
            )

        self.assertEqual(parsed.call_count, 1)
        self.assertEqual(len(model["frames"]), 4)
        # Every frame shares the one parsed pose rather than a private copy.
        first = model["frames"][0]["skeleton"]
        self.assertTrue(all(frame["skeleton"] is first for frame in model["frames"]))

    def test_clip_player_updates_mounted_elements_instead_of_rerunning_streamlit(self) -> None:
        model = _clip_player_model(
            [{"Depth_Color": None, "IR": None, "Thermal": None, "Skeleton": None}],
            {},
            {},
            None,
            None,
            None,
            "test:clip-1",
            {"0.5": 200, "1": 100, "2": 50},
        )

        html = _clip_player_html(model)

        self.assertIn("const imageCache = new Map()", html)
        self.assertIn("window.setTimeout", html)
        self.assertIn("drawFrame(nextFrame)", html)
        self.assertIn("pointermove", html)
        self.assertIn("sessionStorage", html)
        self.assertIn("cuhkx:skeleton-camera:${model.identity}", html)
        # The host iframe is a fixed height, so the player must scroll itself.
        self.assertIn("overflow-y: auto", html)
        # Without radar the multi-megabyte plotly bundle stays out of the page.
        self.assertLess(len(html), 200_000)

    def test_plotly_is_inlined_only_when_the_clip_has_radar(self) -> None:
        common = ([{"Depth_Color": None, "IR": None, "Thermal": None, "Skeleton": None}], {}, {})
        without = _clip_player_html(
            _clip_player_model(*common, None, None, None, "test:clip-1", {})
        )
        with_radar = _clip_player_html(
            _clip_player_model(
                *common, b"frame,x,y,z,v,snr\n7,1,2,3,-0.5,60\n", None, None, "test:clip-1", {}
            )
        )

        self.assertGreater(len(with_radar), len(without) + 500_000)
        # The bundle must not close the script element that wraps it, or the
        # rest of the player would be parsed as markup.
        self.assertEqual(with_radar.count("</script>"), without.count("</script>"))

    def test_image_encoding_shrinks_frames_as_clips_get_longer(self) -> None:
        self.assertEqual(_image_encoding(1), IMAGE_ENCODING_STEPS[0])
        self.assertEqual(_image_encoding(10_000), IMAGE_ENCODING_STEPS[-1])
        edges = [_image_encoding(count)[0] for count in (1, 200, 800, 5_000)]
        self.assertEqual(edges, sorted(edges, reverse=True))

    def test_frames_are_reencoded_and_the_resize_bound_binds(self) -> None:
        """The production path re-encodes; the fallback keeps the original bytes.

        Depth Color and IR are already 640x480, so the first encoding step only
        changes the format. The later steps have to actually shrink the pixels,
        which is what keeps a long clip's payload bounded.
        """

        from PIL import Image

        buffer = io.BytesIO()
        Image.frombytes(
            "RGB",
            (640, 480),
            bytes((index // 3) % 256 for index in range(640 * 480 * 3)),
        ).save(buffer, format="PNG")
        original = buffer.getvalue()

        def decoded(uri: str | None) -> Image.Image:
            assert uri is not None
            self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
            return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))

        full = _image_data_uri("frame.png", original, 640, 82)
        self.assertEqual(decoded(full).size, (640, 480))

        for max_edge, quality in IMAGE_ENCODING_STEPS[1:]:
            reduced = _image_data_uri("frame.png", original, max_edge, quality)
            self.assertLessEqual(max(decoded(reduced).size), max_edge)
            assert reduced is not None and full is not None
            self.assertLess(len(reduced), len(full))

    def test_undecodable_frames_fall_back_only_for_browser_formats(self) -> None:
        self.assertTrue(
            (_image_data_uri("frame.png", b"not an image", 640, 82) or "").startswith(
                "data:image/png;base64,"
            )
        )
        # An unknown suffix would embed a URI no browser can render.
        self.assertIsNone(_image_data_uri("frame.raw", b"not an image", 640, 82))

    def test_clip_median_lengths_stabilize_each_frame(self) -> None:
        first_points = [[float(index), 0.0, float(index * 2)] for index in range(17)]
        second_points = [[float(index * 2), 0.0, float(index * 4)] for index in range(17)]
        payloads = [
            json.dumps([{"keypoints": points, "keypoint_scores": [1.0] * 17}]).encode()
            for points in (first_points, second_points)
        ]

        _, bone_lengths = skeleton_calibration(payloads)

        self.assertIsNotNone(bone_lengths)
        assert bone_lengths is not None
        for points in (first_points, second_points):
            stabilized = _stabilize_skeleton_points(
                [tuple(point) for point in points],
                bone_lengths,
            )
            for expected, (start, end) in zip(bone_lengths, SKELETON_EDGES, strict=True):
                self.assertAlmostEqual(math.dist(stabilized[start], stabilized[end]), expected)

    def test_stabilized_pose_keeps_every_joint_attached(self) -> None:
        """Bone lengths alone cannot detect a scrambled skeleton.

        Stabilization walks outward from the pelvis, so a joint whose parent has
        not been placed yet is built from the origin instead. Every bone would
        still measure the right length, so only checking positions catches it.
        """

        points = [(float(index), float(index % 4), float(index * 2)) for index in range(17)]
        payload = json.dumps(
            [{"keypoints": [list(point) for point in points], "keypoint_scores": [1.0] * 17}]
        ).encode()
        _, bone_lengths = skeleton_calibration([payload])
        assert bone_lengths is not None

        stabilized = _stabilize_skeleton_points(points, bone_lengths)

        # Lengths are the clip medians of a single frame, so the pose is
        # reproduced exactly rather than merely having correct bone lengths.
        for original, rebuilt in zip(points, stabilized, strict=True):
            self.assertAlmostEqual(math.dist(original, rebuilt), 0.0, places=6)

    def test_edge_chain_is_a_tree_rooted_at_the_pelvis(self) -> None:
        self.assertEqual(
            SKELETON_EDGES,
            (
                (0, 1), (1, 2), (2, 3),
                (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (9, 10),
                (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16),
            ),
        )
        chain = _skeleton_edge_chain(SKELETON_EDGES)
        self.assertEqual(len(chain), len(SKELETON_EDGES))
        placed = {0}
        for start, end, edge_index in chain:
            self.assertIn(start, placed, "parent joint must be placed before its child")
            placed.add(end)
            self.assertEqual(SKELETON_EDGES[edge_index], (start, end))
        self.assertEqual(placed, set(range(len(SKELETON_JOINT_NAMES))))

    def test_edge_chain_rejects_a_disconnected_edge_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "one tree rooted at joint 0"):
            _skeleton_edge_chain(((0, 1), (5, 6)))
        with self.assertRaisesRegex(ValueError, "more than one parent"):
            _skeleton_edge_chain(((0, 1), (0, 2), (1, 2)))

    def test_reordered_edges_still_rebuild_the_pose(self) -> None:
        """Reordering SKELETON_EDGES must not scramble the rendered skeleton."""

        points = [(float(index), float(index % 3), float(index * 2)) for index in range(17)]
        payload = json.dumps(
            [{"keypoints": [list(point) for point in points], "keypoint_scores": [1.0] * 17}]
        ).encode()
        _, bone_lengths = skeleton_calibration([payload])
        assert bone_lengths is not None
        expected = _stabilize_skeleton_points(points, bone_lengths)

        # Children before parents — the order that used to build from the origin.
        reversed_edges = tuple(reversed(SKELETON_EDGES))
        reversed_lengths = tuple(reversed(bone_lengths))
        with patch("visualization.app.SKELETON_EDGE_CHAIN", _skeleton_edge_chain(reversed_edges)):
            rebuilt = _stabilize_skeleton_points(points, reversed_lengths)
        for first, second in zip(expected, rebuilt, strict=True):
            self.assertAlmostEqual(math.dist(first, second), 0.0, places=6)


class DiscoverModelArtifactsTests(unittest.TestCase):
    def test_returns_empty_when_no_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(discover_model_artifacts(tmp), [])

    def test_discovers_and_sorts_newest_first(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts" / "logreg").mkdir(parents=True)
            (root / "artifacts" / "late_fusion").mkdir(parents=True)
            older = root / "artifacts" / "logreg" / "model.joblib"
            newer = root / "artifacts" / "late_fusion" / "model.joblib"
            older.write_bytes(b"a")
            newer.write_bytes(b"b")
            os.utime(older, ns=(1_000_000_000, 1_000_000_000))
            os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
            result = discover_model_artifacts(root)
            self.assertEqual(result[0], newer)
            self.assertEqual(result[1], older)

    def test_ignores_non_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts" / "logreg"
            root.mkdir(parents=True)
            (root / "validation.json").write_text("{}", encoding="utf-8")
            self.assertEqual(discover_model_artifacts(tmp), [])


@needs_numpy
class PredictionTableFromArraysTests(unittest.TestCase):
    def test_builds_table_from_arrays(self) -> None:
        mapping = {0: "0_Wash_face", 1: "1_Brush_teeth"}
        table = _prediction_table_from_arrays(
            ["SM_test_0001", "SM_test_0002"],
            [0, 1],
            ["a/SM_test_0001", "a/SM_test_0002"],
            mapping,
        )
        self.assertEqual(len(table.by_clip), 2)
        self.assertEqual(table.by_clip["SM_test_0001"].action_name, "0_Wash_face")
        self.assertEqual(table.rows_read, 2)

    def test_rejects_unknown_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown action_id"):
            _prediction_table_from_arrays(["SM_test_0001"], [99], ["a"], {0: "0_Wash_face"})

    def test_rejects_duplicate_clip(self) -> None:
        mapping = {0: "0_Wash_face"}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _prediction_table_from_arrays(
                ["SM_test_0001", "SM_test_0001"], [0, 0], ["a", "b"], mapping
            )


@needs_numpy
class GeneratePredictionsMockedTests(unittest.TestCase):
    def _setup_mocks(self, tmp_path: Path):
        # Create class_mapping
        mapping_path = tmp_path / "Training" / "class_mapping.csv"
        mapping_path.parent.mkdir(parents=True)
        mapping_path.write_text("action_id,action_name\n0,0_Wash_face\n1,1_Brush_teeth\n", encoding="utf-8")
        mapping = {0: "0_Wash_face", 1: "1_Brush_teeth"}
        # Fake records
        mock_train = [MagicMock(clip_id="0_Wash_face/user1/1-1-1"), MagicMock(clip_id="1_Brush_teeth/user1/1-1-2")]
        mock_test = [
            MagicMock(clip_id="SM_test_0001", submission_path="small_model_track_test/SM_test_0001"),
            MagicMock(clip_id="SM_test_0002", submission_path="small_model_track_test/SM_test_0002"),
        ]

        class FakeBundle:
            def __init__(self, clip_ids, submission_paths=None):
                import numpy as np

                self.clip_ids = np.array(clip_ids)
                self.submission_paths = np.array(submission_paths) if submission_paths else None

        import numpy as np

        train_bundle = FakeBundle(["0_Wash_face/user1/1-1-1", "1_Brush_teeth/user1/1-1-2"])
        test_bundle = FakeBundle(["SM_test_0001", "SM_test_0002"], ["small_model_track_test/SM_test_0001", "small_model_track_test/SM_test_0002"])
        mock_model = MagicMock()
        mock_model.feature_config = {}
        mock_model.reducers = {}
        mock_model.predict.side_effect = [np.array([0, 1]), np.array([1, 0])]
        return mapping, mock_train, mock_test, train_bundle, test_bundle, mock_model

    def test_generate_train_and_test(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            mapping, mock_train, mock_test, train_bundle, test_bundle, mock_model = self._setup_mocks(repo)
            model_path = repo / "artifacts" / "logreg" / "model.joblib"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"dummy")

            def fake_train(root):
                return mock_train, MagicMock()

            def fake_test(root, csv=None):
                return mock_test

            fake_config = MagicMock(return_value=MagicMock())
            def fake_extract(records, config, n_jobs=None, progress_callback=None):
                if records == mock_train:
                    if progress_callback:
                        progress_callback(1, 2)
                        progress_callback(2, 2)
                    return train_bundle
                if progress_callback:
                    progress_callback(1, 2)
                    progress_callback(2, 2)
                return test_bundle

            with patch("visualization.predictions._require_modeling_stack") as mock_req:
                mock_req.return_value = (fake_train, fake_test, fake_config, fake_extract, MagicMock(return_value=mock_model))
                with patch("visualization.dataset.resolve_dataset_root", return_value=repo):
                    train_table = generate_predictions_from_model(model_path, dataset_root=repo, action_mapping=mapping, split="train", n_jobs=1)
                    self.assertEqual(train_table.by_clip["0_Wash_face/user1/1-1-1"].action_id, 0)
                    self.assertEqual(train_table.by_clip["1_Brush_teeth/user1/1-1-2"].action_id, 1)

                    mock_model.predict = MagicMock(return_value=np.array([1, 0]))
                    mock_req.return_value = (fake_train, fake_test, fake_config, fake_extract, MagicMock(return_value=mock_model))
                    test_table = generate_predictions_from_model(model_path, dataset_root=repo, action_mapping=mapping, split="test", n_jobs=1)
                    self.assertEqual(test_table.by_clip["SM_test_0001"].action_id, 1)

    def test_generate_all_splits(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            mapping, mock_train, mock_test, train_bundle, test_bundle, mock_model = self._setup_mocks(repo)
            model_path = repo / "artifacts" / "logreg" / "model.joblib"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"dummy")

            def fake_train(root):
                return mock_train, MagicMock()

            def fake_test(root, csv=None):
                return mock_test

            fake_config = MagicMock(return_value=MagicMock())
            def fake_extract(records, config, n_jobs=None, progress_callback=None):
                return train_bundle if records == mock_train else test_bundle

            mock_model.predict = MagicMock(side_effect=[np.array([0, 1]), np.array([1, 0])])
            with patch("visualization.predictions._require_modeling_stack") as mock_req:
                mock_req.return_value = (fake_train, fake_test, fake_config, fake_extract, MagicMock(return_value=mock_model))
                with patch("visualization.dataset.resolve_dataset_root", return_value=repo):
                    all_tables = generate_all_split_predictions(model_path, dataset_root=repo, action_mapping=mapping, n_jobs=1)
                    self.assertIn("train", all_tables)
                    self.assertIn("test", all_tables)
                    self.assertEqual(len(all_tables["train"].by_clip), 2)
                    self.assertEqual(len(all_tables["test"].by_clip), 2)

    def test_generate_with_progress_callback(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            mapping, mock_train, mock_test, train_bundle, test_bundle, mock_model = self._setup_mocks(repo)
            model_path = repo / "artifacts" / "logreg" / "model.joblib"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"dummy")

            def fake_train(root):
                return mock_train, MagicMock()

            def fake_test(root, csv=None):
                return mock_test

            fake_config = MagicMock(return_value=MagicMock())
            calls: list[tuple[int, int]] = []

            def fake_extract(records, config, n_jobs=None, progress_callback=None):
                if progress_callback:
                    # Report fine-grained progress so an early tick in the test
                    # phase would land below any milestone emitted after train.
                    for done in range(1, 11):
                        progress_callback(done, 10)
                    calls.append((10, 10))
                return train_bundle if records == mock_train else test_bundle

            mock_model.predict = MagicMock(side_effect=[np.array([0, 1]), np.array([1, 0])])
            with patch("visualization.predictions._require_modeling_stack") as mock_req:
                mock_req.return_value = (fake_train, fake_test, fake_config, fake_extract, MagicMock(return_value=mock_model))
                with patch("visualization.dataset.resolve_dataset_root", return_value=repo):
                    progress_calls: list[tuple[int, int, str]] = []

                    def outer_cb(pct: int, total: int, msg: str) -> None:
                        progress_calls.append((pct, total, msg))

                    all_tables = generate_all_split_predictions(
                        model_path, dataset_root=repo, action_mapping=mapping, n_jobs=1, progress_callback=outer_cb
                    )
                    # Should have received progress updates for both splits
                    self.assertGreater(len(progress_calls), 0)
                    self.assertIn("train", all_tables)
                    # Every update is a 0-100 percentage that never moves backwards.
                    percents = [pct for pct, _total, _msg in progress_calls]
                    self.assertTrue(all(total == 100 for _pct, total, _msg in progress_calls))
                    self.assertEqual(percents, sorted(percents))
                    self.assertTrue(all(0 <= pct <= 100 for pct in percents))
                    # Both phases are labelled, and every update carries a message.
                    messages = " ".join(msg for _pct, _total, msg in progress_calls)
                    self.assertIn("training features", messages)
                    self.assertIn("test features", messages)
                    self.assertTrue(all(msg for _pct, _total, msg in progress_calls))

    def test_rejects_invalid_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "artifacts" / "logreg" / "model.joblib").parent.mkdir(parents=True)
            (repo / "artifacts" / "logreg" / "model.joblib").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "split must be"):
                generate_predictions_from_model(repo / "artifacts" / "logreg" / "model.joblib", split="invalid", action_mapping={0: "a"})


class SavePredictionCsvTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = Path(tmp) / "classes.csv"
            mapping_path.write_text("action_id,action_name\n0,0_Wash_face\n1,1_Brush_teeth\n", encoding="utf-8")
            mapping = load_action_mapping(mapping_path)
            table = PredictionTable(
                by_clip={
                    "SM_test_0001": ClipPrediction("SM_test_0001", 0, "0_Wash_face", "a/SM_test_0001"),
                    "SM_test_0002": ClipPrediction("SM_test_0002", 1, "1_Brush_teeth", "a/SM_test_0002"),
                },
                rows_read=2,
                blank_predictions=0,
            )
            out = Path(tmp) / "out.csv"
            saved = save_prediction_csv(table, out)
            self.assertTrue(saved.is_file())
            # Verify CSV content
            with saved.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
            # And load back
            from visualization.predictions import load_prediction_csv

            loaded = load_prediction_csv(saved, mapping)
            self.assertEqual(loaded.by_clip["SM_test_0001"].action_id, 0)

    def test_csv_text_matches_saved_file(self) -> None:
        """The sidebar download and the on-disk writer must not drift apart."""

        table = PredictionTable(
            by_clip={
                "SM_test_0001": ClipPrediction("SM_test_0001", 0, "0_Wash_face", "a/SM_test_0001"),
                "SM_test_0002": ClipPrediction("SM_test_0002", 1, "1_Brush_teeth", "a/SM_test_0002"),
            },
            rows_read=2,
            blank_predictions=0,
        )
        text = prediction_csv_text(table)
        self.assertEqual(
            text.splitlines(),
            ["path,prediction", "a/SM_test_0001,0", "a/SM_test_0002,1"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = save_prediction_csv(table, Path(tmp) / "nested" / "out.csv")
            # Compare bytes: csv writes RFC 4180 CRLF terminators, which
            # ``read_text`` would silently translate away.
            self.assertEqual(out.read_bytes(), text.encode("utf-8"))
            self.assertIn(b"\r\n", out.read_bytes())

    def test_combined_train_and_test_csv_round_trips_full_clip_ids(self) -> None:
        mapping = {0: "0_Wash_face", 1: "1_Brush_teeth"}
        train_id = "0_Wash_face/user1/1-1-1"
        train = PredictionTable(
            by_clip={
                train_id: ClipPrediction(train_id, 0, "0_Wash_face", train_id)
            },
            rows_read=1,
            blank_predictions=0,
        )
        test = PredictionTable(
            by_clip={
                "SM_test_0001": ClipPrediction(
                    "SM_test_0001",
                    1,
                    "1_Brush_teeth",
                    "small_model_track_test/SM_test_0001/",
                )
            },
            rows_read=1,
            blank_predictions=0,
        )
        combined = combine_prediction_tables({"train": train, "test": test})

        with tempfile.TemporaryDirectory() as tmp:
            output = save_prediction_csv(
                combined,
                Path(tmp) / "both.csv",
                identifier="clip_id",
            )
            from visualization.predictions import load_prediction_csv

            loaded = load_prediction_csv(output, mapping)

        self.assertEqual(set(loaded.by_clip), {train_id, "SM_test_0001"})


@needs_app
class AddPredictionsTrainTestTests(unittest.TestCase):
    def test_add_predictions_supports_train_and_test(self) -> None:
        frame = pd.DataFrame(
            [
                {"split": "train", "clip_id": "0_Wash_face/user1/1-1-1", "action_id": 0, "action_name": "0_Wash_face"},
                {"split": "train", "clip_id": "1_Brush_teeth/user2/1-1-2", "action_id": 1, "action_name": "1_Brush_teeth"},
                {"split": "test", "clip_id": "SM_test_0001", "action_id": None, "action_name": None},
                {"split": "test", "clip_id": "SM_test_0002", "action_id": None, "action_name": None},
            ]
        )
        train_table = PredictionTable(
            by_clip={
                "0_Wash_face/user1/1-1-1": ClipPrediction("0_Wash_face/user1/1-1-1", 0, "0_Wash_face", "0_Wash_face/user1/1-1-1"),
                "1_Brush_teeth/user2/1-1-2": ClipPrediction("1_Brush_teeth/user2/1-1-2", 0, "0_Wash_face", "1_Brush_teeth/user2/1-1-2"),
            },
            rows_read=2,
            blank_predictions=0,
        )
        test_table = PredictionTable(
            by_clip={
                "SM_test_0001": ClipPrediction("SM_test_0001", 0, "0_Wash_face", "small_model_track_test/SM_test_0001"),
                "SM_test_0002": ClipPrediction("SM_test_0002", 1, "1_Brush_teeth", "small_model_track_test/SM_test_0002"),
            },
            rows_read=2,
            blank_predictions=0,
        )
        combined = PredictionTable(by_clip={**train_table.by_clip, **test_table.by_clip}, rows_read=4, blank_predictions=0)
        enriched = add_predictions(frame, combined)
        # Check train correctness: first correct, second incorrect
        self.assertTrue(enriched.loc[enriched["clip_id"] == "0_Wash_face/user1/1-1-1", "prediction_correct"].iloc[0])
        self.assertFalse(enriched.loc[enriched["clip_id"] == "1_Brush_teeth/user2/1-1-2", "prediction_correct"].iloc[0])
        # Test rows should be <NA>
        self.assertTrue(pd.isna(enriched.loc[enriched["clip_id"] == "SM_test_0001", "prediction_correct"].iloc[0]))
        # Test that train with no prediction would be <NA> not False
        frame2 = pd.DataFrame(
            [
                {"split": "train", "clip_id": "0_Wash_face/user1/1-1-1", "action_id": 0, "action_name": "0_Wash_face"},
                {"split": "train", "clip_id": "0_Wash_face/user1/1-1-3", "action_id": 0, "action_name": "0_Wash_face"},
            ]
        )
        # Only first has prediction
        partial = PredictionTable(
            by_clip={"0_Wash_face/user1/1-1-1": ClipPrediction("0_Wash_face/user1/1-1-1", 0, "0_Wash_face", "x")},
            rows_read=1,
            blank_predictions=0,
        )
        enriched2 = add_predictions(frame2, partial)
        self.assertTrue(pd.isna(enriched2.loc[enriched2["clip_id"] == "0_Wash_face/user1/1-1-3", "prediction_correct"].iloc[0]))



@needs_modeling
class ExtractFeatureBundleProgressTests(unittest.TestCase):
    def test_progress_callback_is_invoked(self) -> None:
        # Use real extract_feature_bundle with mocked _extract_job to avoid heavy I/O
        from modeling.features import FeatureConfig, extract_feature_bundle
        from modeling.data import ClipRecord
        from pathlib import Path

        # Create dummy ClipRecords (paths not needed because we mock the job)
        records = [
            ClipRecord(split="train", clip_id=f"0_Wash_face/user1/{i}", depth_dir=Path("/tmp"), ir_dir=Path("/tmp"), imu_dir=Path("/tmp"), skeleton_dir=Path("/tmp"), label=0, user="user1")
            for i in range(3)
        ]
        config = FeatureConfig()

        # Patch _extract_job to return dummy feature arrays quickly
        import numpy as np
        from modeling.features import image_feature_size, imu_feature_size, skeleton_feature_size, image_engineered_feature_size, imu_engineered_feature_size, skeleton_engineered_feature_size

        def fake_extract(job):
            _record, cfg = job
            return (
                np.zeros(image_feature_size(cfg), dtype=np.float32),
                np.zeros(image_feature_size(cfg), dtype=np.float32) if False else None,  # ir not used in this config
                np.zeros(imu_feature_size(cfg), dtype=np.float32),
                np.zeros(skeleton_feature_size(cfg), dtype=np.float32),
                np.zeros(image_engineered_feature_size(cfg, include_color=True), dtype=np.float32),
                np.zeros(image_engineered_feature_size(cfg, include_color=False), dtype=np.float32),
                np.zeros(imu_engineered_feature_size(cfg), dtype=np.float32),
                np.zeros(skeleton_engineered_feature_size(cfg), dtype=np.float32),
            )

        # Need to handle the actual signature: _extract_job takes (ClipRecord, FeatureConfig)
        # Our fake should ignore and return the tuple above. Use more precise patch.
        with patch("modeling.features._extract_job", side_effect=fake_extract):
            calls: list[tuple[int, int]] = []

            def cb(done, total):
                calls.append((done, total))

            # progress_every=1 to get a callback per clip
            bundle = extract_feature_bundle(records, config, n_jobs=1, progress_every=1, progress_callback=cb)
            # Should have been called 3 times (once per clip, plus final)
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[-1], (3, 3))
            self.assertEqual(bundle.clip_ids.tolist(), [r.clip_id for r in records])

    def test_callback_errors_are_not_swallowed(self) -> None:
        """A wrong-arity callback must fail loudly, not silently stop updating."""

        from modeling.data import ClipRecord
        from modeling.features import FeatureConfig, extract_feature_bundle

        records = [
            ClipRecord(
                split="train",
                clip_id="0_Wash_face/user1/1",
                depth_dir=Path("/tmp"),
                ir_dir=Path("/tmp"),
                imu_dir=Path("/tmp"),
                skeleton_dir=Path("/tmp"),
                label=0,
                user="user1",
            )
        ]

        def broken_callback(done: int) -> None:  # wrong arity
            raise AssertionError("should not be reached")

        with patch("modeling.features._extract_job", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                extract_feature_bundle(records, FeatureConfig(), n_jobs=1)
        with self.assertRaises(TypeError):
            with patch("modeling.features._extract_job", side_effect=self._dummy_job):
                extract_feature_bundle(
                    records,
                    FeatureConfig(),
                    n_jobs=1,
                    progress_every=1,
                    progress_callback=broken_callback,
                )

    @staticmethod
    def _dummy_job(job: object) -> tuple[object, ...]:
        import numpy as np
        from modeling.features import (
            image_engineered_feature_size,
            image_feature_size,
            imu_engineered_feature_size,
            imu_feature_size,
            skeleton_engineered_feature_size,
            skeleton_feature_size,
        )

        _record, cfg = job
        return (
            np.zeros(image_feature_size(cfg), dtype=np.float32),
            None,
            np.zeros(imu_feature_size(cfg), dtype=np.float32),
            np.zeros(skeleton_feature_size(cfg), dtype=np.float32),
            np.zeros(image_engineered_feature_size(cfg, include_color=True), dtype=np.float32),
            np.zeros(image_engineered_feature_size(cfg, include_color=False), dtype=np.float32),
            np.zeros(imu_engineered_feature_size(cfg), dtype=np.float32),
            np.zeros(skeleton_engineered_feature_size(cfg), dtype=np.float32),
        )




@needs_app
class CorrectnessFlagTests(unittest.TestCase):
    """``prediction_correct`` uses pandas' nullable boolean dtype.

    Scalar access yields ``numpy.bool_``, so ``value is True`` never matched and
    the ✓/✗ markers never rendered.
    """

    def test_normalizes_numpy_bool(self) -> None:
        frame = pd.DataFrame({"prediction_correct": pd.array([True, False, None], dtype="boolean")})
        values = [correctness_flag(frame.iloc[i].get("prediction_correct")) for i in range(3)]
        self.assertEqual(values, [True, False, None])
        for value in values[:2]:
            self.assertIsInstance(value, bool)

    def test_handles_missing_and_plain_python_values(self) -> None:
        self.assertIsNone(correctness_flag(None))
        self.assertIsNone(correctness_flag(pd.NA))
        self.assertTrue(correctness_flag(True))
        self.assertFalse(correctness_flag(False))

    def test_end_to_end_flag_from_add_predictions(self) -> None:
        frame = pd.DataFrame(
            [
                {"split": "train", "clip_id": "a/u/1", "action_id": 0, "action_name": "0_Wash_face"},
                {"split": "train", "clip_id": "b/u/1", "action_id": 1, "action_name": "1_Brush_teeth"},
            ]
        )
        table = PredictionTable(
            by_clip={
                "a/u/1": ClipPrediction("a/u/1", 0, "0_Wash_face", "x"),
                "b/u/1": ClipPrediction("b/u/1", 0, "0_Wash_face", "y"),
            },
            rows_read=2,
            blank_predictions=0,
        )
        enriched = add_predictions(frame, table)
        flags = [correctness_flag(enriched.iloc[i].get("prediction_correct")) for i in range(2)]
        self.assertEqual(flags, [True, False])


if __name__ == "__main__":
    unittest.main()


@needs_app
class PredictionPickerOptionTests(unittest.TestCase):
    """The Workflow hand-off has regressed three times in this picker."""

    def test_candidates_are_listed_newest_first_after_the_none_entry(self) -> None:
        options, _default = prediction_file_options(
            [Path("/repo/outputs/new.csv"), Path("/repo/outputs/old.csv")],
            "",
            Path("/repo"),
        )
        self.assertEqual(
            options, ["", "/repo/outputs/new.csv", "/repo/outputs/old.csv"]
        )

    def test_an_empty_handoff_selects_none_rather_than_the_newest_file(self) -> None:
        """Switching dataset clears the hand-off on purpose.

        Falling back to the newest discovered CSV there would resurrect
        predictions computed against the dataset the user just left.
        """

        _options, default = prediction_file_options(
            [Path("/repo/outputs/new.csv")], "", Path("/repo")
        )
        self.assertEqual(default, "")

    def test_a_handoff_wins_over_the_newest_candidate(self) -> None:
        root = Path("/repo")
        _options, default = prediction_file_options(
            [Path("/repo/outputs/new.csv"), Path("/repo/outputs/old.csv")],
            "/repo/outputs/old.csv",
            root,
        )
        self.assertEqual(default, "/repo/outputs/old.csv")

    def test_a_relative_handoff_is_resolved_against_the_repository(self) -> None:
        # Two candidates, and the hand-off names the *second*. Without
        # resolution it matches nothing and the newest file wins instead, so
        # this only passes when the relative path is joined to the root.
        _options, default = prediction_file_options(
            [Path("/repo/outputs/new.csv"), Path("/repo/outputs/old.csv")],
            "outputs/old.csv",
            Path("/repo"),
        )
        self.assertEqual(default, "/repo/outputs/old.csv")

    def test_a_handoff_outside_outputs_is_offered_when_it_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            elsewhere = root / "artifacts" / "run" / "preds.csv"
            elsewhere.parent.mkdir(parents=True)
            elsewhere.write_text("clip_id,prediction\n", encoding="utf-8")

            options, default = prediction_file_options(
                [root / "outputs" / "a.csv"], str(elsewhere), root
            )

        # Discovery only scans outputs/, but Workflow may write anywhere.
        self.assertIn(str(elsewhere), options)
        self.assertEqual(default, str(elsewhere))

    def test_a_handoff_that_does_not_exist_falls_back_to_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options, default = prediction_file_options(
                [root / "outputs" / "a.csv"], str(root / "gone.csv"), root
            )

        self.assertNotIn(str(root / "gone.csv"), options)
        self.assertEqual(default, str(root / "outputs" / "a.csv"))

    def test_no_candidates_leaves_only_the_disabled_option(self) -> None:
        options, default = prediction_file_options([], "", Path("/repo"))
        self.assertEqual(options, [""])
        self.assertEqual(default, "")
