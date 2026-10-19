"""
Unit tests for PoseClassifier with ONNX Runtime.
"""

import pytest
import numpy as np
from src.engine.classifier import PoseClassifier, POSE_LABELS
from tools.train_classifier import create_base_landmarks
from src.vision.normalizer import LandmarkNormalizer


def test_classifier_loads_and_infers():
    clf = PoseClassifier(model_path="data/models/pose_classifier.onnx", min_probability=0.70)
    assert clf.session is not None

    normalizer = LandmarkNormalizer()
    base_templates = create_base_landmarks()

    for expected_pose, raw_lm in base_templates.items():
        _, feat_63d, _ = normalizer.normalize(raw_lm)
        label, prob = clf.predict(feat_63d)
        assert label == expected_pose
        assert prob >= 0.70
