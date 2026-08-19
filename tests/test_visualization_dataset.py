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
    build_member_index,
    build_source_manifest,
    parse_member,
    resolve_dataset_root,
)
from visualization.playback import normalized_timeline_position, playback_interval, timeline_frame_count


class DatasetParsingTests(unittest.TestCase):
    def test_playback_timeline_helpers(self) -> None:
        modalities = {
            "Depth_Color": ["depth-1", "depth-2", "depth-3"],
            "Thermal": [f"thermal-{index}" for index in range(8)],
        }
        self.assertEqual(timeline_frame_count(modalities), 3)
        self.assertEqual(normalized_timeline_position(1, 3), 50)
        self.assertEqual(normalized_timeline_position(0, 1), 0)
        self.assertAlmostEqual(playback_interval(2.0, 3, 1.0), 1.0)
        self.assertAlmostEqual(playback_interval(2.0, 3, 2.0), 0.5)

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
