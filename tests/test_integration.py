"""
Integration test: small end-to-end pipeline using synthetic data.
No real LFW images or TFDS are loaded — everything is synthetic.
Tests that the full evaluation flow runs without errors and produces
sensible outputs.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    compute_scores,
    threshold_sweep,
    select_threshold,
    evaluate_at_threshold,
    compute_auc,
)
from src.tracking import RunTracker
from src.validation import validate_pairs_df, validate_features_coverage


def _make_synthetic_data(n_pos=20, n_neg=20, dim=64, seed=42):
    """
    Create synthetic pairs and features.
    Positive pairs share the same base vector + small noise (high similarity).
    Negative pairs come from different random vectors (low similarity).
    """
    rng = np.random.default_rng(seed)

    features = {}
    pair_rows = []

    # Positive pairs: 20 identities, 2 images each
    for k in range(n_pos):
        base = rng.standard_normal(dim)
        base /= np.linalg.norm(base)
        for j in range(2):
            noise = rng.standard_normal(dim) * 0.05
            vec = base + noise
            vec /= np.linalg.norm(vec)
            features[f"pos_id{k}_img{j}"] = vec.astype(np.float32)
        pair_rows.append({
            "left_path": f"pos_id{k}_img0",
            "right_path": f"pos_id{k}_img1",
            "label": 1,
            "split": "test",
        })

    # Negative pairs: 20 random cross-identity pairs
    ids = [f"pos_id{k}_img0" for k in range(n_pos)]
    for _ in range(n_neg):
        i, j = rng.choice(n_pos, size=2, replace=False)
        pair_rows.append({
            "left_path": ids[i],
            "right_path": ids[j],
            "label": 0,
            "split": "test",
        })

    pairs_df = pd.DataFrame(pair_rows)
    return pairs_df, features


def test_full_evaluation_pipeline():
    """End-to-end: create pairs + features → validate → score → sweep → select → log."""
    pairs_df, features = _make_synthetic_data(n_pos=30, n_neg=30)

    # Validation
    validate_pairs_df(pairs_df)
    validate_features_coverage(features, pairs_df)

    # Scoring
    scores, labels = compute_scores(pairs_df, features)
    assert len(scores) == len(pairs_df)
    assert scores.min() >= -1.0 and scores.max() <= 1.01  # cosine range

    # Threshold sweep
    thresholds = np.linspace(0.0, 1.0, 51)
    sweep = threshold_sweep(scores, labels, thresholds)
    assert len(sweep) == 51
    assert all("balanced_accuracy" in r for r in sweep)

    # Select threshold
    best_thresh, best_metrics = select_threshold(sweep, "maximize_balanced_accuracy")
    assert 0.0 <= best_thresh <= 1.0
    assert 0.0 <= best_metrics["balanced_accuracy"] <= 1.0

    # Evaluate at selected threshold
    final_metrics = evaluate_at_threshold(scores, labels, best_thresh)
    assert final_metrics["TP"] + final_metrics["FN"] == (labels == 1).sum()
    assert final_metrics["TN"] + final_metrics["FP"] == (labels == 0).sum()

    # AUC is computable
    auc = compute_auc(sweep)
    assert 0.0 <= auc <= 1.0

    # Positive pairs should score higher than negatives on average
    assert scores[labels == 1].mean() > scores[labels == 0].mean(), \
        "Synthetic positive pairs should have higher cosine similarity than negatives"


def test_run_tracking(tmp_path):
    """RunTracker correctly writes and reads run entries."""
    runs_file = str(tmp_path / "runs.json")
    tracker = RunTracker(runs_file)

    tracker.log_run(
        run_id="test_run_001",
        config_name="m2.yaml",
        split="val",
        data_version="baseline",
        threshold=0.72,
        metrics={"accuracy": 0.85, "FAR": 0.12, "FRR": 0.18},
        artifacts=["outputs/eval/plots/roc.png"],
        note="Integration test run",
    )

    # Verify it was saved
    assert os.path.exists(runs_file)
    run = tracker.get_run("test_run_001")
    assert run["threshold"] == 0.72
    assert run["metrics"]["accuracy"] == 0.85
    assert tracker.get_selected_threshold("test_run_001") == 0.72

    # Overwrite same run_id
    tracker.log_run(
        run_id="test_run_001",
        config_name="m2.yaml",
        split="val",
        data_version="baseline",
        threshold=0.80,
        metrics={"accuracy": 0.87},
        artifacts=[],
        note="Updated",
    )
    run = tracker.get_run("test_run_001")
    assert run["threshold"] == 0.80

    # Non-existent run raises KeyError
    with pytest.raises(KeyError):
        tracker.get_run("does_not_exist")


def test_positive_pairs_higher_similarity():
    """Verify that synthetic positive pairs score higher than negatives."""
    pairs_df, features = _make_synthetic_data(seed=7)
    scores, labels = compute_scores(pairs_df, features)
    pos_mean = scores[labels == 1].mean()
    neg_mean = scores[labels == 0].mean()
    assert pos_mean > neg_mean, \
        f"Expected pos mean ({pos_mean:.3f}) > neg mean ({neg_mean:.3f})"
