from __future__ import annotations

import csv
import json
import tempfile
import unittest
import warnings
from pathlib import Path

try:
    import numpy as np
    from PIL import Image

    from modeling.algorithms import available_algorithms, get_algorithm
    from modeling.cache import load_feature_cache, save_feature_cache
    from modeling.data import EXPECTED_IMU_DEVICES, discover_training_clips
    from modeling.model import (
        FittedMultimodalModel,
        library_versions,
        load_model,
        save_model,
    )
    from modeling.features import (
        FeatureConfig,
        RawFeatureBundle,
        extract_image_feature_pair,
        extract_image_features,
        extract_imu_feature_pair,
        extract_imu_features,
        extract_skeleton_feature_pair,
        extract_skeleton_features,
        image_engineered_feature_size,
        image_feature_size,
        imu_engineered_feature_size,
        imu_feature_size,
        skeleton_engineered_feature_size,
        skeleton_feature_size,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the extras
    raise unittest.SkipTest(
        f"modeling tests require modeling/requirements.txt ({exc})"
    ) from exc


IMU_HEADER = [
    "time",
    "DeviceName",
    "AccX(g)",
    "AccY(g)",
    "AccZ(g)",
    "AsX(°/s)",
    "AsY(°/s)",
    "AsZ(°/s)",
    "AngleX(°)",
    "AngleY(°)",
    "AngleZ(°)",
    "MagX(uT)",
    "MagY(uT)",
    "MagZ(uT)",
    "Q0",
    "Q1",
    "Q2",
    "Q3",
]


def write_imu(path: Path, devices: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(IMU_HEADER)
        for index, device in enumerate(devices):
            writer.writerow(
                [
                    f"2025-06-17 12:00:0{1-index}.000",
                    f"{device}(id)",
                    index + 0.1,
                    index + 0.2,
                    index + 0.3,
                    index + 1.1,
                    index + 1.2,
                    index + 1.3,
                    index * 10.0,
                    index * 5.0,
                    index * 2.0,
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )


def skeleton_person(offset: float) -> dict[str, object]:
    points = [[offset + joint * 0.01, joint * 0.02, 1.0 + joint * 0.01] for joint in range(17)]
    points[5] = [offset - 0.2, 0.5, 1.0]
    points[6] = [offset + 0.2, 0.5, 1.0]
    points[11] = [offset - 0.1, 0.0, 1.0]
    points[12] = [offset + 0.1, 0.0, 1.0]
    return {"keypoints": points, "keypoint_scores": [1.0] * 17}


class FeatureExtractionTests(unittest.TestCase):
    def test_image_features_have_fixed_size(self) -> None:
        config = FeatureConfig()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in range(3):
                array = np.zeros((48, 64), dtype=np.uint8)
                array[:, index * 8 : index * 8 + 16] = 80 + index * 50
                Image.fromarray(array).save(directory / f"IR_2025-01-01_00-00-0{index}.000_{index}.png")
            features = extract_image_features(directory, config)
        self.assertEqual(features.shape, (image_feature_size(config),))
        self.assertTrue(np.isfinite(features).all())

    def test_color_image_engineered_features_have_fixed_size(self) -> None:
        config = FeatureConfig()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in range(4):
                array = np.zeros((48, 64, 3), dtype=np.uint8)
                array[:, : 16 + index * 8, index % 3] = 80 + index * 40
                Image.fromarray(array).save(directory / f"Depth_2025-01-01_00-00-0{index}.000_{index}.png")
            base, engineered = extract_image_feature_pair(
                directory,
                config,
                include_color=True,
            )
            omitted_base, engineered_only = extract_image_feature_pair(
                directory,
                config,
                include_color=False,
                include_base=False,
            )
        self.assertEqual(base.shape, (image_feature_size(config),))
        self.assertEqual(
            engineered.shape,
            (image_engineered_feature_size(config, include_color=True),),
        )
        self.assertTrue(np.isfinite(engineered).all())
        self.assertIsNone(omitted_base)
        self.assertEqual(
            engineered_only.shape,
            (image_engineered_feature_size(config, include_color=False),),
        )

    def test_imu_features_cover_five_devices(self) -> None:
        config = FeatureConfig()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_imu(directory / "down.csv", ("WTLL", "WTRL"))
            write_imu(directory / "up.csv", ("WTLA", "WTRA", "WTC"))
            features = extract_imu_features(directory, config)
        self.assertEqual(features.shape, (imu_feature_size(config),))
        self.assertTrue(np.isfinite(features).all())

    def test_imu_engineered_features_have_fixed_size(self) -> None:
        config = FeatureConfig()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_imu(directory / "down.csv", ("WTLL", "WTRL"))
            write_imu(directory / "up.csv", ("WTLA", "WTRA", "WTC"))
            base, engineered = extract_imu_feature_pair(directory, config)
        self.assertEqual(base.shape, (imu_feature_size(config),))
        self.assertEqual(engineered.shape, (imu_engineered_feature_size(config),))
        self.assertTrue(np.isfinite(engineered).all())

    def test_skeleton_features_are_root_centered(self) -> None:
        config = FeatureConfig(skeleton_frames=4)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "predictions"
            directory.mkdir(parents=True)
            for index in range(3):
                payload = [skeleton_person(float(index) * 0.1)]
                (directory / f"Color_2025-01-01_00-00-0{index}.000_{index}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            features = extract_skeleton_features(directory.parent, config)
        self.assertEqual(features.shape, (skeleton_feature_size(config),))
        positions = features[: config.skeleton_frames * 17 * 3].reshape(config.skeleton_frames, 17, 3)
        roots = (positions[:, 11] + positions[:, 12]) / 2.0
        np.testing.assert_allclose(roots, 0.0, atol=1e-5)

    def test_skeleton_engineered_features_have_fixed_size(self) -> None:
        config = FeatureConfig(skeleton_frames=4)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "predictions"
            directory.mkdir(parents=True)
            for index in range(5):
                payload = [skeleton_person(float(index) * 0.1)]
                (directory / f"Color_2025-01-01_00-00-0{index}.000_{index}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            base, engineered = extract_skeleton_feature_pair(directory.parent, config)
        self.assertEqual(base.shape, (skeleton_feature_size(config),))
        self.assertEqual(engineered.shape, (skeleton_engineered_feature_size(config),))
        self.assertTrue(np.isfinite(engineered).all())


class CompleteCaseDiscoveryTests(unittest.TestCase):
    def test_incomplete_imu_device_set_is_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "Training" / "data" / "HAR" / "data"
            for trial in ("1-1-1", "1-1-2"):
                key = Path("0_Wash_face") / "user1" / trial
                for modality in ("Depth_Color", "IR"):
                    directory = data / modality / key
                    directory.mkdir(parents=True)
                    (directory / "frame.png").write_bytes(b"image")
                skeleton = data / "Skeleton" / key / "predictions"
                skeleton.mkdir(parents=True)
                (skeleton / "frame.json").write_text(json.dumps([skeleton_person(0.0)]))
                imu = data / "IMU" / key
                write_imu(imu / "down.csv", ("WTLL", "WTRL"))
                upper = ("WTLA", "WTRA", "WTC") if trial == "1-1-1" else ("WTLA", "WTRA")
                write_imu(imu / "up.csv", upper)

            records, report = discover_training_clips(root)
        self.assertEqual([record.clip_id for record in records], ["0_Wash_face/user1/1-1-1"])
        self.assertEqual(report.modality_complete, 2)
        self.assertEqual(report.all_imu_devices, 1)


class FeatureCacheTests(unittest.TestCase):
    def test_cache_round_trip(self) -> None:
        config = FeatureConfig()
        train = RawFeatureBundle(
            clip_ids=np.asarray(["train"]),
            depth=np.ones((1, 2), dtype=np.float32),
            ir=np.ones((1, 3), dtype=np.float32),
            imu=np.ones((1, 4), dtype=np.float32),
            skeleton=np.ones((1, 5), dtype=np.float32),
            imu_engineered=np.ones((1, 6), dtype=np.float32),
            labels=np.asarray([1]),
            groups=np.asarray(["user1"]),
        )
        test = RawFeatureBundle(
            clip_ids=np.asarray(["test"]),
            depth=np.ones((1, 2), dtype=np.float32),
            ir=np.ones((1, 3), dtype=np.float32),
            imu=np.ones((1, 4), dtype=np.float32),
            skeleton=np.ones((1, 5), dtype=np.float32),
            imu_engineered=np.ones((1, 6), dtype=np.float32),
            submission_paths=np.asarray(["small_model_track_test/test/"]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npz"
            save_feature_cache(path, train, test, config)
            loaded_train, loaded_test = load_feature_cache(path, config)
        np.testing.assert_array_equal(loaded_train.labels, train.labels)
        np.testing.assert_array_equal(loaded_test.submission_paths, test.submission_paths)
        np.testing.assert_array_equal(loaded_train.imu_engineered, train.imu_engineered)


class AlgorithmRegistryTests(unittest.TestCase):
    def test_logistic_regression_is_registered(self) -> None:
        self.assertIn("logistic_regression", available_algorithms())
        algorithm = get_algorithm("logistic_regression")
        self.assertEqual(
            algorithm.parameter_candidates(),
            [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
        )
        estimator = algorithm.build_estimator({"C": 0.25}, random_state=7)
        self.assertEqual(estimator.C, 0.25)
        self.assertEqual(estimator.random_state, 7)

    def test_search_space_can_be_overridden(self) -> None:
        algorithm = get_algorithm("logistic_regression")
        self.assertEqual(
            algorithm.parameter_candidates({"C": [0.5, 2.0]}),
            [{"C": 0.5}, {"C": 2.0}],
        )


class ModelVersionMetadataTests(unittest.TestCase):
    def _model(self, versions: dict[str, str]) -> FittedMultimodalModel:
        return FittedMultimodalModel(
            reducers={},
            combined_scaler=None,
            classifier=None,
            feature_config={},
            pca_components={},
            library_versions=versions,
        )

    def _round_trip(self, versions: dict[str, str]) -> list[warnings.WarningMessage]:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.joblib"
            save_model(self._model(versions), path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_model(path)
        return [item for item in caught if item.category is RuntimeWarning]

    def test_matching_versions_do_not_warn(self) -> None:
        self.assertEqual(self._round_trip(library_versions()), [])

    def test_mismatched_version_warns(self) -> None:
        versions = dict(library_versions())
        versions["scikit-learn"] = "0.0.1-not-installed"
        caught = self._round_trip(versions)
        self.assertEqual(len(caught), 1)
        self.assertIn("scikit-learn 0.0.1-not-installed", str(caught[0].message))

    def test_artifact_without_versions_warns(self) -> None:
        caught = self._round_trip({})
        self.assertEqual(len(caught), 1)
        self.assertIn("records no library versions", str(caught[0].message))

    def test_library_versions_are_recorded(self) -> None:
        versions = library_versions()
        self.assertEqual(set(versions), {"scikit-learn", "numpy"})
        self.assertTrue(all(versions.values()))


if __name__ == "__main__":
    unittest.main()
