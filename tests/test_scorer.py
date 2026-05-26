import numpy as np
import tensorflow as tf

from src.inference.anomaly_scorer import (
    ReconstructionAnomalyScorer,
    classify_scores,
    compute_threshold,
    mahalanobis_scores,
    reconstruction_error,
    score_to_probability,
    summarize_scores,
    threshold_from_config,
)


def test_reconstruction_error_is_per_sample_mse():
    original = np.zeros((2, 2, 2, 1), dtype=np.float32)
    reconstructed = np.array(
        [[[[1.0], [1.0]], [[1.0], [1.0]]], [[[0.0], [2.0]], [[0.0], [2.0]]]],
        dtype=np.float32,
    )

    scores = reconstruction_error(original, reconstructed)

    assert np.allclose(scores, [1.0, 2.0])


def test_threshold_helpers_and_binary_predictions():
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    threshold = compute_threshold(scores, method="percentile", percentile=50)
    fixed = threshold_from_config(scores, {"inference": {"threshold_method": "fixed", "fixed_threshold": 0.25}})

    assert np.isclose(threshold.value, 0.25)
    assert fixed.value == 0.25
    assert classify_scores([0.24, 0.25], fixed.value).tolist() == [0, 1]
    assert np.allclose(score_to_probability([0.125, 0.25], fixed.value), [0.5, 1.0])


def test_summary_and_optional_mahalanobis_score():
    summary = summarize_scores([1.0, 2.0, 3.0])
    scores = mahalanobis_scores(
        np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32),
        mean_vector=np.zeros(2, dtype=np.float32),
        inverse_covariance=np.eye(2, dtype=np.float32),
    )

    assert summary.count == 3
    assert np.allclose(scores, [0.0, 5.0])


def test_autoencoder_reference_dataset_does_not_treat_targets_as_labels():
    inputs = tf.ones((2, 4, 4, 1), dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, inputs)).batch(2)
    model = tf.keras.Sequential([tf.keras.layers.Lambda(lambda values: values)])

    scores, labels = ReconstructionAnomalyScorer(model=model).score_dataset(dataset)

    assert np.allclose(scores, [0.0, 0.0])
    assert labels is None
