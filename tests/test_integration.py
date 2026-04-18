import csv
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import compute_metrics_at_threshold, select_threshold_max_balanced_accuracy
from src.validation import validate_pair_file, validate_scores_match_pairs, assert_valid
from src.tracking import log_run
from src.error_analysis import build_results_df, slice_false_positives, summarize_slice
from src.inference import run_pair_inference, apply_threshold, compute_confidence

import pandas as pd


def _write_synthetic_pairs(path: str, n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(rng.integers(0, 2))
        rows.append({
            "left_path": f"lfw_record_Alice_{i:06d}",
            "right_path": f"lfw_record_Bob_{i:06d}",
            "label": label,
            "split": "val",
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["left_path", "right_path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _mock_scores(pairs_df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = np.where(
        pairs_df["label"].values == 1,
        rng.normal(0.75, 0.15, size=len(pairs_df)),
        rng.normal(0.35, 0.18, size=len(pairs_df)),
    )
    return np.clip(scores, -1.0, 1.0)


def test_full_eval_pipeline(tmp_path):
    pairs_path = str(tmp_path / "val_pairs.csv")
    _write_synthetic_pairs(pairs_path, n=100)

    errors = validate_pair_file(pairs_path)
    assert errors == [], f"Pair file errors: {errors}"

    pairs_df = pd.read_csv(pairs_path)
    pairs_df["label"] = pairs_df["label"].astype(int)

    scores = _mock_scores(pairs_df)

    errors = validate_scores_match_pairs(scores, pairs_df)
    assert errors == []

    thresholds = np.linspace(scores.min(), scores.max(), 50)
    selected_t = select_threshold_max_balanced_accuracy(scores, pairs_df["label"].values, thresholds)
    assert -1.0 <= selected_t <= 1.0

    metrics = compute_metrics_at_threshold(scores, pairs_df["label"].values, selected_t)
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"] == len(pairs_df)

    results_df = build_results_df(pairs_df, scores, selected_t)
    fp_summary = summarize_slice(slice_false_positives(results_df), "false_positives")
    assert "count" in fp_summary

    runs_dir = str(tmp_path / "runs")
    run_id = log_run(
        metrics=metrics,
        threshold=selected_t,
        split="val",
        config_name="test_config.yaml",
        data_version="v1",
        note="integration test",
        runs_dir=runs_dir,
    )
    assert run_id.startswith("run_")

    assert os.path.exists(os.path.join(runs_dir, "runs.jsonl"))
    assert os.path.exists(os.path.join(runs_dir, "runs_summary.csv"))

    with open(os.path.join(runs_dir, "runs.jsonl")) as f:
        record = json.loads(f.readline())
    assert record["run_id"] == run_id
    assert record["split"] == "val"
    assert "balanced_accuracy" in record


def test_inference_smoke_with_synthetic_images(tmp_path):
    rng = np.random.default_rng(0)
    left_img = (rng.integers(0, 256, (160, 160, 3), dtype=np.uint8))
    right_img = (rng.integers(0, 256, (160, 160, 3), dtype=np.uint8))

    result = run_pair_inference(
        left_img,
        right_img,
        threshold=0.35,
        model_backend=("mock", None),
    )

    assert "score" in result
    assert "decision" in result
    assert "confidence" in result
    assert "latency_total_s" in result
    assert result["decision"] in (0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["latency_total_s"] > 0
    assert result["threshold"] == 0.35


def test_inference_decision_consistent_with_score():
    rng = np.random.default_rng(1)
    img_a = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    img_b = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    threshold = 0.35
    result = run_pair_inference(img_a, img_b, threshold=threshold, model_backend=("mock", None))
    expected_decision = 1 if result["score"] >= threshold else 0
    assert result["decision"] == expected_decision


def test_inference_confidence_matches_score():
    rng = np.random.default_rng(2)
    img_a = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    img_b = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    threshold = 0.35
    result = run_pair_inference(img_a, img_b, threshold=threshold, model_backend=("mock", None))
    expected_conf = compute_confidence(result["score"], threshold)
    assert abs(result["confidence"] - expected_conf) < 1e-9


def test_inference_latency_stages_sum_to_total():
    rng = np.random.default_rng(3)
    img_a = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    img_b = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    result = run_pair_inference(img_a, img_b, threshold=0.35, model_backend=("mock", None))
    stage_sum = result["latency_preprocess_s"] + result["latency_embedding_s"] + result["latency_scoring_s"]
    assert stage_sum <= result["latency_total_s"] + 1e-3