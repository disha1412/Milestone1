"""Unit tests for evaluation metric functions in src/evaluation.py."""

import numpy as np
import pytest
from src.evaluation import (
    threshold_sweep,
    select_threshold,
    evaluate_at_threshold,
    compute_auc,
)


def test_evaluate_perfect_predictions():
    """All correct predictions: accuracy=1, FAR=0, FRR=0."""
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1,   1,   0,   0])
    metrics = evaluate_at_threshold(scores, labels, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["FAR"] == 0.0
    assert metrics["FRR"] == 0.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["TP"] == 2
    assert metrics["TN"] == 2
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0


def test_evaluate_all_wrong():
    """All predictions inverted: TP=0, TN=0, all errors."""
    scores = np.array([0.1, 0.1, 0.9, 0.9])
    labels = np.array([1,   1,   0,   0])
    metrics = evaluate_at_threshold(scores, labels, threshold=0.5)
    assert metrics["TP"] == 0
    assert metrics["TN"] == 0
    assert metrics["FP"] == 2
    assert metrics["FN"] == 2
    assert metrics["accuracy"] == 0.0


def test_threshold_sweep_length():
    """Sweep should return one entry per threshold."""
    scores = np.random.rand(100)
    labels = (scores > 0.5).astype(int)
    thresholds = np.linspace(0, 1, 11)
    results = threshold_sweep(scores, labels, thresholds)
    assert len(results) == 11


def test_threshold_sweep_far_frr_tradeoff():
    """
    As threshold increases: FAR should decrease, FRR should increase.
    (Monotonic trend on average — can have ties but overall direction holds.)
    """
    rng = np.random.default_rng(42)
    labels = np.array([1] * 50 + [0] * 50)
    scores = np.where(labels == 1,
                      rng.uniform(0.4, 0.9, 100),
                      rng.uniform(0.1, 0.6, 100))
    thresholds = np.linspace(0.0, 1.0, 21)
    results = threshold_sweep(scores, labels, thresholds)
    fars = [r["FAR"] for r in results]
    frrs = [r["FRR"] for r in results]
    # Overall trend: FAR decreases, FRR increases as threshold increases
    assert fars[0] >= fars[-1], "FAR should decrease as threshold increases"
    assert frrs[0] <= frrs[-1], "FRR should increase as threshold increases"


def test_select_threshold_maximize_balanced_accuracy():
    """Selected threshold should have the highest balanced accuracy in the sweep."""
    rng = np.random.default_rng(0)
    labels = np.array([1] * 50 + [0] * 50)
    scores = np.where(labels == 1, rng.uniform(0.5, 1.0, 100), rng.uniform(0.0, 0.5, 100))
    thresholds = np.linspace(0, 1, 51)
    sweep = threshold_sweep(scores, labels, thresholds)
    best_thresh, best_metrics = select_threshold(sweep, "maximize_balanced_accuracy")
    # Verify the returned metrics equal those in the sweep at best_thresh
    matching = [r for r in sweep if abs(r["threshold"] - best_thresh) < 1e-9]
    assert len(matching) == 1
    assert abs(matching[0]["balanced_accuracy"] - best_metrics["balanced_accuracy"]) < 1e-9


def test_auc_random_classifier():
    """Random classifier should have AUC close to 0.5."""
    rng = np.random.default_rng(123)
    labels = rng.integers(0, 2, 200)
    scores = rng.uniform(0, 1, 200)
    thresholds = np.linspace(0, 1, 101)
    sweep = threshold_sweep(scores, labels, thresholds)
    auc = compute_auc(sweep)
    assert 0.3 <= auc <= 0.7, f"Random AUC should be near 0.5, got {auc:.3f}"


def test_auc_perfect_classifier():
    """Perfect classifier should have AUC close to 1.0."""
    labels = np.array([1] * 50 + [0] * 50)
    scores = np.array([0.9] * 50 + [0.1] * 50)
    thresholds = np.linspace(0, 1, 101)
    sweep = threshold_sweep(scores, labels, thresholds)
    auc = compute_auc(sweep)
    assert auc > 0.95, f"Perfect classifier AUC should be ~1.0, got {auc:.3f}"


def test_metrics_all_same_score():
    """Edge case: all scores equal to threshold → stable behavior, no exceptions."""
    scores = np.full(10, 0.5)
    labels = np.array([1, 0] * 5)
    metrics = evaluate_at_threshold(scores, labels, threshold=0.5)
    assert "accuracy" in metrics
    assert "FAR" in metrics
