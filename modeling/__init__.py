"""Classical multimodal baselines for the CUHK-X dataset."""

from modeling.algorithms import available_algorithms, get_algorithm
from modeling.data import ClipRecord, EXPECTED_IMU_DEVICES
from modeling.features import FeatureConfig, RawFeatureBundle

__all__ = [
    "ClipRecord",
    "EXPECTED_IMU_DEVICES",
    "FeatureConfig",
    "RawFeatureBundle",
    "available_algorithms",
    "get_algorithm",
]
