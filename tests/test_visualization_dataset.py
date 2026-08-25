from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from visualization.dataset import (
    DATASET_ROOT_ENV_VAR,
    DEFAULT_DATASET_ROOT,
    DataSource,
    Member,
    build_clip_member_index,
    build_dataset_manifest,
    build_initial_dataset_manifest,
    build_member_index,
    build_source_manifest,
    initial_clip_ids,
    parse_member,
    resolve_dataset_root,
    write_manifest,
)
from visualization.playback import (
    playback_interval,
    timeline_frame_count,
    timeline_member_index,
)


class DatasetParsingTests(unittest.TestCase):
    def test_playback_timeline_helpers(self) -> None:
        modalities = {
            "Depth_Color": ["depth-1", "depth-2", "depth-3"],
            "Thermal": [f"thermal-{index}" for index in range(8)],
        }
        self.assertEqual(timeline_frame_count(modalities), 3)
        self.assertAlmostEqual(playback_interval(2.0, 3, 1.0), 1.0)
        self.assertAlmostEqual(playback_interval(2.0, 3, 2.0), 0.5)

    def test_timeline_member_mapping_does_not_quantize_through_percent(self) -> None:
        self.assertEqual(timeline_member_index(567, 1001, 1001), 567)
        self.assertEqual(timeline_member_index(4, 9, 26), 12)
        self.assertEqual(timeline_member_index(20, 9, 3), 2)
        self.assertEqual(timeline_member_index(0, 1, 0), 0)

    def test_parse_training_member(self) -> None:
        source = DataSource("train", "zip", "/tmp/HAR_full.zip")
        parsed = parse_member(
            source,
            Member("HAR/data/IR/6_Drink_water/user2/1-1-1/IR_2025-01-02_03-04-05.600_0001.png", 10),
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.action_id, 6)
        self.assertEqual(parsed.clip_id, "6_Drink_water/user2/1-1-1")
        self.assertEqual(parsed.timestamp, "2025-01-02_03-04-05.600")

    def test_zip_manifest_and_quality_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "test.zip"
            prefix = "small_model_track_test/SM_test_0001"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"{prefix}/Depth_Color/Depth_2025-01-02_03-04-05.000_0001_Color.png", b"png")
                archive.writestr(f"{prefix}/IR/IR_2025-01-02_03-04-05.000_0001.png", b"png")
                archive.writestr(f"{prefix}/Thermal/frame_000001.jpg", b"jpg")
                archive.writestr(f"{prefix}/IMU/down.csv", "time,device,ax\n")
                archive.writestr(f"{prefix}/Radar/radar.csv", "timestamp,frame,x,y,z,v,snr,noise\n")
                archive.writestr(
                    f"{prefix}/Skeleton/predictions/Color_2025-01-02_03-04-05.000_0001.json",
                    json.dumps([]),
                )

            source = DataSource("test", "zip", str(archive_path))
            records = build_source_manifest(source, deep=True)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertTrue(record["complete"])
            self.assertTrue(record["radar_empty"])
            self.assertEqual(record["imu_empty_files"], 1)
            self.assertEqual(record["skeleton_empty_frames"], 1)
            self.assertIn("radar has no detections", record["issues"])

            index = build_member_index(source)
            self.assertEqual(len(index["SM_test_0001"]["Depth_Color"]), 1)

            # Per-clip index must match the full index for that clip
            clip_index = build_clip_member_index(source, "SM_test_0001")
            self.assertEqual(clip_index["Depth_Color"], index["SM_test_0001"]["Depth_Color"])
            self.assertEqual(clip_index["IR"], index["SM_test_0001"]["IR"])


class ClipMemberIndexTests(unittest.TestCase):
    def test_per_clip_matches_full_for_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Build a minimal train directory structure
            root = Path(tmp) / "HAR"
            for modality in ("Depth_Color", "IR"):
                d = root / "data" / modality / "0_Wash_face" / "user1" / "1-1-1"
                d.mkdir(parents=True)
                (d / f"{modality}_2025-01-02_03-04-05.000_0001.png").write_bytes(b"png")
                # Second clip that must NOT appear in per-clip result
                d2 = root / "data" / modality / "1_Brush_teeth" / "user2" / "1-1-2"
                d2.mkdir(parents=True)
                (d2 / f"{modality}_2025-01-02_03-04-06.000_0001.png").write_bytes(b"png")
            source = DataSource("train", "directory", str(root))
            full = build_member_index(source)
            self.assertEqual(len(full), 2)
            clip = build_clip_member_index(source, "0_Wash_face/user1/1-1-1")
            self.assertEqual(set(clip.keys()), {"Depth_Color", "IR"})
            self.assertEqual(len(clip["Depth_Color"]), 1)
            # Must match the full index entry
            self.assertEqual(clip["Depth_Color"], full["0_Wash_face/user1/1-1-1"]["Depth_Color"])

    def test_per_clip_for_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "test.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("small_model_track_test/SM_test_0001/IR/file.png", b"png")
                archive.writestr("small_model_track_test/SM_test_0002/IR/file.png", b"png")
            source = DataSource("test", "zip", str(archive_path))
            clip = build_clip_member_index(source, "SM_test_0001")
            self.assertEqual(len(clip["IR"]), 1)
            self.assertIn("SM_test_0001", clip["IR"][0])

    def test_per_clip_missing_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "small_model_track_test"
            root.mkdir(parents=True)
            source = DataSource("test", "directory", str(root))
            self.assertEqual(build_clip_member_index(source, "SM_test_9999"), {})


class ProgressiveManifestTests(unittest.TestCase):
    def test_initial_training_clips_are_spread_across_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HAR"
            for action in ("0_Wash_face", "1_Brush_teeth"):
                for trial in ("1-1-1", "1-1-2", "1-1-3"):
                    clip = root / "data" / "Depth_Color" / action / "user1" / trial
                    clip.mkdir(parents=True)
                    (clip / "frame.png").write_bytes(b"png")
            source = DataSource("train", "directory", str(root))
            selected = initial_clip_ids(source, 2)
            self.assertEqual(len(selected), 2)
            self.assertEqual(
                {clip.split("/", 1)[0] for clip in selected},
                {"0_Wash_face", "1_Brush_teeth"},
            )

    def test_initial_manifest_skips_archive_splits_but_still_reports_them(self) -> None:
        """An archived split contributes no preview clips yet stays discoverable.

        The dashboard names such a split in its partial-data banner, so
        ``sources`` must keep it even though ``records`` cannot cover it.
        """

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            train_data = dataset_root / "Training" / "data"
            train_data.mkdir(parents=True)
            with zipfile.ZipFile(train_data / "HAR_full.zip", "w") as archive:
                archive.writestr("HAR/data/Depth_Color/0_Action/user1/1-1-1/frame.png", b"png")
            test_clip = (
                dataset_root / "Testing" / "data" / "small_model_track_test" / "SM_test_0001" / "Depth_Color"
            )
            test_clip.mkdir(parents=True)
            (test_clip / "frame.png").write_bytes(b"png")

            records, sources = build_initial_dataset_manifest(dataset_root, max_clips=4)
            self.assertEqual(sources["train"].kind, "zip")
            self.assertEqual({record["split"] for record in records}, {"test"})
            self.assertEqual(sorted(sources), ["test", "train"])

    def test_initial_manifest_splits_budget_between_train_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            train_root = dataset_root / "Training" / "data" / "HAR"
            test_root = dataset_root / "Testing" / "data" / "small_model_track_test"
            for index in range(4):
                train_clip = (
                    train_root
                    / "data"
                    / "Depth_Color"
                    / f"{index}_Action"
                    / "user1"
                    / "1-1-1"
                )
                train_clip.mkdir(parents=True)
                (train_clip / "frame.png").write_bytes(b"png")
                test_clip = test_root / f"SM_test_{index:04d}" / "Depth_Color"
                test_clip.mkdir(parents=True)
                (test_clip / "frame.png").write_bytes(b"png")

            records, sources = build_initial_dataset_manifest(dataset_root, max_clips=5)
            split_counts = {split: sum(record["split"] == split for record in records) for split in sources}
            self.assertEqual(len(records), 5)
            self.assertEqual(split_counts, {"train": 3, "test": 2})

    def test_manifest_write_replaces_target_without_leaving_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "manifest.json"
            output.write_text("old", encoding="utf-8")
            write_manifest([{"clip_id": "new"}], output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), [{"clip_id": "new"}])
            self.assertEqual(list(output.parent.glob(".*.tmp.json")), [])

    def test_full_manifest_reports_exact_file_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            train_clip = (
                dataset_root
                / "Training"
                / "data"
                / "HAR"
                / "data"
                / "Depth_Color"
                / "0_Action"
                / "user1"
                / "1-1-1"
            )
            train_clip.mkdir(parents=True)
            (train_clip / "frame.png").write_bytes(b"png")
            test_clip = (
                dataset_root
                / "Testing"
                / "data"
                / "small_model_track_test"
                / "SM_test_0001"
                / "IR"
            )
            test_clip.mkdir(parents=True)
            (test_clip / "frame.png").write_bytes(b"png")

            updates: list[tuple[str, int, int]] = []
            records, _ = build_dataset_manifest(
                dataset_root,
                deep_test=False,
                progress_callback=lambda phase, processed, total: updates.append(
                    (phase, processed, total)
                ),
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(updates[0], ("counting", 0, 0))
            self.assertEqual(updates[-1], ("test", 2, 2))


class DatasetRootResolutionTests(unittest.TestCase):
    def test_explicit_value_wins_over_environment(self) -> None:
        with mock.patch.dict(os.environ, {DATASET_ROOT_ENV_VAR: "/from/env"}):
            self.assertEqual(resolve_dataset_root("/explicit"), Path("/explicit"))

    def test_environment_used_when_no_explicit_value(self) -> None:
        with mock.patch.dict(os.environ, {DATASET_ROOT_ENV_VAR: "/from/env"}):
            self.assertEqual(resolve_dataset_root(), Path("/from/env"))
            self.assertEqual(resolve_dataset_root(None), Path("/from/env"))
            self.assertEqual(resolve_dataset_root(""), Path("/from/env"))

    def test_default_used_when_environment_absent(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_dataset_root(), DEFAULT_DATASET_ROOT)

    def test_user_home_is_expanded(self) -> None:
        with mock.patch.dict(os.environ, {DATASET_ROOT_ENV_VAR: "~/datasets/cuhkx"}):
            resolved = resolve_dataset_root()
        self.assertEqual(resolved, Path.home() / "datasets" / "cuhkx")
        self.assertNotIn("~", str(resolved))


if __name__ == "__main__":
    unittest.main()
