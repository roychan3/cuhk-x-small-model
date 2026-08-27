from __future__ import annotations

import csv
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from PIL import Image

    from modeling.algorithms import available_algorithms, get_algorithm
    from modeling.cache import load_feature_cache, save_feature_cache
    from modeling.data import ClipRecord, EXPECTED_IMU_DEVICES, discover_training_clips
    from modeling.model import (
        FittedMultimodalModel,
        _fit_reducers,
        feature_config_from_artifact,
        _majority_vote,
        cross_validate_detailed,
        library_versions,
        load_model,
        save_model,
        save_validation_outputs,
    )
    from modeling.features import (
        ALL_FEATURE_BLOCKS,
        DEFAULT_FEATURE_BLOCKS,
        HEAD,
        L_ANKLE,
        L_HIP,
        L_SHOULDER,
        L_WRIST,
        PELVIS,
        R_ANKLE,
        R_HIP,
        R_SHOULDER,
        R_WRIST,
        SKELETON_JOINTS,
        THORAX,
        FeatureConfig,
        RawFeatureBundle,
        feature_config_dict,
        extract_image_feature_pair,
        extract_image_features,
        extract_imu_feature_pair,
        extract_imu_features,
        extract_skeleton_feature_pair,
        extract_feature_bundle,
        extract_skeleton_features,
        filter_bundle_to_blocks,
        image_engineered_feature_size,
        image_feature_size,
        imu_engineered_feature_size,
        imu_feature_size,
        normalize_feature_blocks,
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


def skeleton_person(offset: float, wrists_at_head: bool = False) -> dict[str, object]:
    """An anatomically plausible Human3.6M pose, standing and facing +y.

    The hips are deliberately asymmetric about the pelvis so that centring on
    joint 0 and centring on the hip midpoint give different answers; a
    symmetric fixture passes either way and pins nothing.
    """

    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    points[0] = [0.00, 0.0, 0.50]  # Pelvis
    points[1] = [0.12, 0.0, 0.50]  # Right hip  (asymmetric on purpose)
    points[2] = [0.12, 0.0, 0.28]  # Right knee
    points[3] = [0.12, 0.0, 0.05]  # Right ankle
    points[4] = [-0.08, 0.0, 0.50]  # Left hip
    points[5] = [-0.08, 0.0, 0.28]  # Left knee
    points[6] = [-0.08, 0.0, 0.05]  # Left ankle
    points[7] = [0.00, 0.0, 0.70]  # Spine
    points[8] = [0.00, 0.0, 0.90]  # Thorax
    points[9] = [0.00, 0.0, 1.00]  # Neck
    points[10] = [0.00, 0.0, 1.15]  # Head
    points[11] = [-0.18, 0.0, 0.90]  # Left shoulder
    points[12] = [-0.20, 0.0, 0.68]  # Left elbow
    points[13] = [-0.22, 0.0, 0.46]  # Left wrist
    points[14] = [0.18, 0.0, 0.90]  # Right shoulder
    points[15] = [0.20, 0.0, 0.68]  # Right elbow
    points[16] = [0.22, 0.0, 0.46]  # Right wrist
    if wrists_at_head:
        # Both hands raised to the face, as in Wash_face or Brush_teeth.
        points[12] = [-0.16, 0.10, 1.00]
        points[13] = [-0.05, 0.06, 1.13]
        points[15] = [0.16, 0.10, 1.00]
        points[16] = [0.05, 0.06, 1.13]
    for point in points:
        point[0] += offset
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
        # Joint 0 is the Human3.6M root, so it lands on the origin. The hip
        # midpoint must not, or the fixture would pass under the old COCO
        # indexing that treated joints 11 and 12 as the hips.
        np.testing.assert_allclose(positions[:, PELVIS], 0.0, atol=1e-5)
        hip_midpoints = (positions[:, L_HIP] + positions[:, R_HIP]) / 2.0
        self.assertGreater(float(np.abs(hip_midpoints).max()), 1e-3)

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

    def _engineered(self, wrists_at_head: bool) -> np.ndarray:
        config = FeatureConfig(skeleton_frames=4)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "predictions"
            directory.mkdir(parents=True)
            for index in range(5):
                payload = [skeleton_person(float(index) * 0.1, wrists_at_head=wrists_at_head)]
                (directory / f"Color_2025-01-01_00-00-0{index}.000_{index}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            return extract_skeleton_feature_pair(directory.parent, config)[1]

    def test_hand_to_head_distance_responds_to_raising_the_hands(self) -> None:
        """Pins the joint indices to their anatomical meaning.

        Under the previous COCO indexing this pair resolved to neck-to-pelvis,
        a near-rigid trunk length that barely moves when the hands do, so the
        two poses produced almost identical features.
        """

        config = FeatureConfig(skeleton_frames=4)
        signal_size = 10 + config.temporal_bins * 2
        # geometry order: 8 angles, then the 7 distance pairs.
        start = (8 + 0) * signal_size
        lowered = self._engineered(False)[start : start + signal_size]
        raised = self._engineered(True)[start : start + signal_size]

        # Channel 0 of _signal_features is the mean of the signal.
        self.assertGreater(float(lowered[0]), 2.0 * float(raised[0]))

    def test_joint_constants_match_the_dataset_layout(self) -> None:
        points = np.asarray(skeleton_person(0.0)["keypoints"], dtype=np.float32)
        # Head highest, ankles lowest, pelvis between the two.
        self.assertEqual(int(np.argmax(points[:, 2])), HEAD)
        self.assertIn(int(np.argmin(points[:, 2])), (L_ANKLE, R_ANKLE))
        self.assertLess(points[L_ANKLE][2], points[PELVIS][2])
        self.assertLess(points[PELVIS][2], points[THORAX][2])
        self.assertLess(points[THORAX][2], points[HEAD][2])
        # Left joints sit on the opposite side of the body from their mirror.
        for left, right in ((L_HIP, R_HIP), (L_SHOULDER, R_SHOULDER), (L_WRIST, R_WRIST)):
            self.assertLess(points[left][0], points[right][0])
        self.assertEqual(SKELETON_JOINTS, len(points))


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

    def test_same_feature_comparison_algorithms_are_registered(self) -> None:
        self.assertEqual(available_algorithms(), ("late_fusion", "logistic_regression"))

    def test_late_fusion_predicts_probabilities_from_named_groups(self) -> None:
        features = np.asarray(
            [
                [-2.0, -1.0, -2.0, -1.0],
                [-1.5, -1.0, -1.5, -1.0],
                [-1.0, -0.5, -1.0, -0.5],
                [-0.5, -0.5, -0.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 0.5, 1.0, 0.5],
                [1.5, 1.0, 1.5, 1.0],
                [2.0, 1.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        classifier = get_algorithm("late_fusion").fit_estimator(
            features,
            labels,
            {"C": 1.0, "weighting": "equal"},
            7,
            {
                "depth": np.asarray([0, 1]),
                "imu": np.asarray([2, 3]),
            },
        )
        probabilities = classifier.predict_proba(features)
        self.assertEqual(probabilities.shape, (8, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(classifier.predict(features), labels)

        stacked = get_algorithm("late_fusion").fit_estimator(
            features,
            labels,
            {},
            7,
            {
                "depth": np.asarray([0, 1]),
                "imu": np.asarray([2, 3]),
            },
        )
        self.assertIn("combined", stacked.estimators)
        self.assertNotIn(
            "combined",
            stacked.feature_groups,
            "fit() must not write the combined block back into the caller's groups",
        )

    def test_late_fusion_build_estimator_rejects_missing_groups(self) -> None:
        with self.assertRaises(NotImplementedError):
            get_algorithm("late_fusion").build_estimator({"C": 1.0}, 7)

    def test_late_fusion_rejects_a_group_the_profile_cannot_weight(self) -> None:
        features = np.repeat(np.asarray([[-1.0, 1.0]]), 8, axis=0)
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        with self.assertRaises(ValueError) as caught:
            get_algorithm("late_fusion").fit_estimator(
                features,
                labels,
                {"weighting": "equal"},
                7,
                {"thermal": np.asarray([0, 1])},
            )
        self.assertIn("thermal", str(caught.exception))


class CrossValidationOutputTests(unittest.TestCase):
    def test_pca_components_are_capped_for_small_training_sets(self) -> None:
        rng = np.random.default_rng(4)
        arrays = {
            "depth": rng.normal(size=(4, 20)).astype(np.float32),
            "depth_engineered": rng.normal(size=(4, 6)).astype(np.float32),
        }
        # Patch the grid so the assertion pins the clamp, not the tuned values.
        with patch.dict("modeling.model.PCA_COMPONENTS", {"depth": 64}, clear=True):
            reducers = _fit_reducers(arrays, np.arange(4), random_state=2)

        # 64 requested, but only 4 training rows are available to fit.
        self.assertEqual(reducers["depth"].named_steps["pca"].n_components_, 4)
        # Engineered blocks are passed through rather than projected.
        self.assertNotIn("pca", reducers["depth_engineered"].named_steps)

    def test_repeated_grouped_validation_saves_reusable_oof_outputs(self) -> None:
        rng = np.random.default_rng(11)
        labels = np.tile(np.asarray([0, 1, 0, 1]), 4)
        groups = np.repeat(np.asarray(["u1", "u2", "u3", "u4"]), 4)

        def block(width: int) -> np.ndarray:
            signal = labels[:, None] * 2.0 - 1.0
            return np.asarray(
                signal + rng.normal(0.0, 0.1, size=(len(labels), width)),
                dtype=np.float32,
            )

        bundle = RawFeatureBundle(
            clip_ids=np.asarray([f"clip-{index}" for index in range(len(labels))]),
            depth=block(8),
            ir=None,
            imu=block(8),
            skeleton=block(8),
            depth_engineered=block(4),
            ir_engineered=block(4),
            imu_engineered=block(4),
            skeleton_engineered=block(4),
            labels=labels,
            groups=groups,
        )
        components = {"depth": 2, "imu": 2, "skeleton": 2}
        progress: list[tuple[int, int]] = []
        with patch.dict("modeling.model.PCA_COMPONENTS", components, clear=True):
            selected, report, outputs = cross_validate_detailed(
                bundle,
                get_algorithm("logistic_regression"),
                [{"C": 0.1}],
                n_splits=2,
                n_repeats=2,
                random_state=5,
                progress_callback=lambda done, total: progress.append((done, total)),
            )

        self.assertEqual(selected, {"C": 0.1})
        self.assertEqual(report["cv_repeats"], 2)
        self.assertEqual(len(report["folds"]), 4)
        self.assertEqual(len(report["per_class_metrics"]), 2)
        self.assertEqual(len(report["per_group_metrics"]), 4)
        self.assertEqual(report["classes"], [0, 1])
        self.assertEqual(report["metrics_basis"], "mean_of_repeats")

        # The headline metric must describe one model, like the one
        # fit_final_model() ships, not the two-model repeat ensemble.
        candidate = report["aggregate_metrics"][0]
        for name, value in candidate["metrics"].items():
            expected = float(
                np.mean([repeat[name] for repeat in candidate["repeat_metrics"]])
            )
            self.assertAlmostEqual(value, expected, places=12)
        self.assertEqual(set(candidate["consensus_metrics"]), set(candidate["metrics"]))
        self.assertEqual(outputs.fold_assignments.shape, (2, len(labels)))
        self.assertTrue((outputs.fold_assignments > 0).all())
        self.assertEqual(outputs.repeat_predictions.shape, (2, len(labels)))
        self.assertIsNotNone(outputs.probabilities)
        self.assertEqual(outputs.probabilities.shape, (len(labels), 2))
        self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oof_predictions.npz"
            save_validation_outputs(outputs, path)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["clip_ids"], bundle.clip_ids)
                np.testing.assert_array_equal(saved["labels"], labels)
                self.assertEqual(saved["probabilities"].shape, (len(labels), 2))

    def test_artifact_config_refuses_a_pre_h36m_model(self) -> None:
        """Artifacts predating the skeleton fix must fail, not predict quietly.

        Their skeleton columns were built by indexing Human3.6M poses as COCO,
        so the current extractor produces something they were never fitted on.
        """

        model = FittedMultimodalModel(
            reducers={},
            combined_scaler=None,
            classifier=None,
            feature_config={"skeleton_frames": 32},
            pca_components={},
        )
        with self.assertRaisesRegex(ValueError, "predates the skeleton joint-order"):
            feature_config_from_artifact(model)

    def test_artifact_config_round_trips_a_current_model(self) -> None:
        config = FeatureConfig(skeleton_frames=8, temporal_bins=3)
        model = FittedMultimodalModel(
            reducers={},
            combined_scaler=None,
            classifier=None,
            feature_config=feature_config_dict(config),
            pca_components={},
        )
        self.assertEqual(feature_config_from_artifact(model), config)

    def test_artifact_config_still_infers_the_legacy_ir_block(self) -> None:
        values = feature_config_dict(FeatureConfig())
        del values["include_legacy_ir"]
        model = FittedMultimodalModel(
            reducers={"ir": object()},
            combined_scaler=None,
            classifier=None,
            feature_config=values,
            pca_components={},
        )
        self.assertTrue(feature_config_from_artifact(model).include_legacy_ir)

    def test_majority_vote_does_not_always_break_ties_towards_the_low_class(self) -> None:
        # Two repeats that disagree on every clip: argmax would hand every clip
        # to class 3, the lower id.
        predictions = np.asarray([[7] * 200, [3] * 200])
        classes = np.asarray([3, 7])
        consensus = _majority_vote(predictions, classes, random_state=5)

        self.assertEqual(set(np.unique(consensus)), {3, 7})
        share = float((consensus == 3).mean())
        self.assertGreater(share, 0.3)
        self.assertLess(share, 0.7)
        np.testing.assert_array_equal(
            consensus,
            _majority_vote(predictions, classes, random_state=5),
        )

    def test_majority_vote_respects_an_actual_majority(self) -> None:
        predictions = np.asarray([[7, 3], [3, 3], [3, 7], [3, 3]])
        np.testing.assert_array_equal(
            _majority_vote(predictions, np.asarray([3, 7]), random_state=0),
            np.asarray([3, 3]),
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


class FeatureBlockSelectionTests(unittest.TestCase):
    def test_blocks_are_canonically_ordered_so_the_same_run_matches_itself(self) -> None:
        first = normalize_feature_blocks(["skeleton", "depth"])
        second = normalize_feature_blocks(["depth", "skeleton"])

        # The tuple lands in validation.json and is compared against caches, so
        # request order must not create two identities for one ablation.
        self.assertEqual(first, second)
        self.assertEqual(first, ("depth", "skeleton"))

    def test_unknown_and_empty_selections_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown feature block"):
            normalize_feature_blocks(["depth", "lidar"])
        with self.assertRaisesRegex(ValueError, "At least one"):
            normalize_feature_blocks([])
        self.assertEqual(normalize_feature_blocks(None), DEFAULT_FEATURE_BLOCKS)

    def test_default_omits_only_the_legacy_ir_base(self) -> None:
        self.assertEqual(
            set(ALL_FEATURE_BLOCKS) - set(DEFAULT_FEATURE_BLOCKS), {"ir"}
        )

    def test_a_missing_row_is_rejected_instead_of_shortening_a_column(self) -> None:
        """A gap would pair later clips' features with the wrong labels."""

        records = [
            ClipRecord(
                split="train",
                clip_id=f"0_Wash_face/user1/{index}",
                depth_dir=Path("/tmp"),
                ir_dir=Path("/tmp"),
                imu_dir=Path("/tmp"),
                skeleton_dir=Path("/tmp"),
                label=0,
                user="user1",
            )
            for index in range(3)
        ]
        config = FeatureConfig()
        blocks = ("depth", "imu")

        def job(args: tuple[object, ...]) -> tuple[object, ...]:
            record, cfg, _selected = args
            drop = record.clip_id.endswith("1")
            values = {
                "depth": None if drop else np.zeros(image_feature_size(cfg), dtype=np.float32),
                "imu": np.zeros(imu_feature_size(cfg), dtype=np.float32),
            }
            return tuple(values.get(name) for name in ALL_FEATURE_BLOCKS)

        with patch("modeling.features._extract_job", side_effect=job):
            with self.assertRaisesRegex(ValueError, "produced no features for clip"):
                extract_feature_bundle(
                    records, config, n_jobs=1, selected_blocks=blocks
                )

    def test_filtering_keeps_only_the_requested_blocks(self) -> None:
        rows = 4
        bundle = RawFeatureBundle(
            clip_ids=np.asarray([f"c{i}" for i in range(rows)]),
            depth=np.zeros((rows, 3), dtype=np.float32),
            imu=np.zeros((rows, 5), dtype=np.float32),
            skeleton=np.zeros((rows, 7), dtype=np.float32),
            labels=np.zeros(rows, dtype=np.int64),
            groups=np.asarray(["u"] * rows),
        )

        filtered = filter_bundle_to_blocks(bundle, ["depth", "skeleton"])

        self.assertEqual(set(filtered.modality_arrays()), {"depth", "skeleton"})
        # Identity columns must survive the filter.
        np.testing.assert_array_equal(filtered.clip_ids, bundle.clip_ids)
        np.testing.assert_array_equal(filtered.labels, bundle.labels)
