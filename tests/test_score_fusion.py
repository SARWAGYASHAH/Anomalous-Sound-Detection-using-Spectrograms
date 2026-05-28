import numpy as np
import pandas as pd

from src.utils.score_fusion import align_score_frames, normalize_scores, sweep_weighted_fusion


def test_align_score_frames_by_row_order_when_filepaths_are_absent():
    left = pd.DataFrame({"label": [0, 1], "score": [0.1, 0.8]})
    right = pd.DataFrame({"label": [0, 1], "score": [2.0, 3.0]})

    aligned = align_score_frames(left, right, left_name="v2", right_name="stgram")

    assert aligned["label"].tolist() == [0, 1]
    assert aligned["v2_score"].tolist() == [0.1, 0.8]
    assert aligned["stgram_score"].tolist() == [2.0, 3.0]


def test_rank_normalization_preserves_order():
    normalized = normalize_scores(np.asarray([10.0, 30.0, 20.0]), method="rank")

    assert normalized.tolist() == [0.0, 1.0, 0.5]


def test_sweep_weighted_fusion_finds_better_weight():
    labels = np.asarray([0, 0, 1, 1])
    weak_scores = np.asarray([0.4, 0.6, 0.5, 0.7])
    strong_scores = np.asarray([0.1, 0.2, 0.8, 0.9])

    best_weight, best_auc, _, sweep = sweep_weighted_fusion(labels, weak_scores, strong_scores, step=0.5)

    assert best_weight == 0.0
    assert best_auc == 1.0
    assert len(sweep) == 3
