import numpy as np
import pytest

from src.inference.classifier import (
    SeverityClassifier,
    classify_score,
    classify_scores,
    classify_scores_as_levels,
)


def test_classify_score_uses_higher_bucket_at_boundaries():
    assert classify_score(0.49) == "normal"
    assert classify_score(0.50) == "follow_up"
    assert classify_score(0.79) == "follow_up"
    assert classify_score(0.80) == "alert"


def test_classify_scores_returns_labels_for_batch():
    labels = classify_scores(np.array([0.1, 0.5, 0.8], dtype=np.float32))

    assert labels.tolist() == ["normal", "follow_up", "alert"]


def test_classify_scores_as_levels_returns_numeric_severity():
    levels = classify_scores_as_levels([0.1, 0.5, 0.8])

    assert levels.tolist() == [0, 1, 2]


def test_classify_scores_as_levels_supports_custom_boundary_count():
    levels = classify_scores_as_levels([1.0, 2.0, 3.0, 4.0], boundaries=[2.0, 3.0, 4.0])

    assert levels.tolist() == [0, 1, 2, 3]


def test_predict_results_include_attention_flags():
    classifier = SeverityClassifier()
    results = classifier.predict_results([0.1, 0.7])

    assert results[0].label == "normal"
    assert results[0].is_anomaly is False
    assert results[1].label == "follow_up"
    assert results[1].requires_attention is True


def test_classifier_can_be_built_from_project_config():
    config = {
        "inference": {
            "classification": {
                "labels": ["ok", "watch", "stop"],
                "boundaries": [10.0, 20.0],
            }
        }
    }

    classifier = SeverityClassifier.from_config(config)

    assert classifier.predict([9.0, 10.0, 20.0]).tolist() == ["ok", "watch", "stop"]


def test_invalid_boundaries_raise_clear_error():
    with pytest.raises(ValueError, match="strictly increasing"):
        SeverityClassifier(boundaries=[0.8, 0.5])


def test_invalid_label_count_raises_clear_error():
    with pytest.raises(ValueError, match="len\\(boundaries\\) \\+ 1"):
        SeverityClassifier(boundaries=[0.5, 0.8], labels=["normal", "alert"])
