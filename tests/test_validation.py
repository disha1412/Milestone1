"""Unit tests for src/validation.py."""

import numpy as np
import pandas as pd
import pytest
from src.validation import validate_pairs_df, validate_features_coverage, validate_metrics


def _make_valid_df():
    return pd.DataFrame({
        "left_path":  ["lfw_record_Alice_000001", "lfw_record_Bob_000010"],
        "right_path": ["lfw_record_Alice_000002", "lfw_record_Carol_000020"],
        "label":      [1, 0],
        "split":      ["val", "val"],
    })


# ── validate_pairs_df ────────────────────────────────────────────────────────

def test_valid_pairs_pass():
    validate_pairs_df(_make_valid_df())  # should not raise


def test_missing_required_column():
    df = _make_valid_df().drop(columns=["label"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_pairs_df(df)


def test_empty_dataframe():
    df = pd.DataFrame(columns=["left_path", "right_path", "label"])
    with pytest.raises(ValueError, match="empty"):
        validate_pairs_df(df)


def test_null_values_rejected():
    df = _make_valid_df()
    df.loc[0, "left_path"] = None
    with pytest.raises(ValueError, match="null"):
        validate_pairs_df(df)


def test_invalid_label_value():
    df = _make_valid_df()
    df.loc[0, "label"] = 2  # invalid
    with pytest.raises(ValueError, match="Invalid label"):
        validate_pairs_df(df)


def test_all_positive_no_negative():
    df = _make_valid_df()
    df["label"] = 1  # no negatives
    with pytest.raises(ValueError, match="no negative pairs"):
        validate_pairs_df(df)


def test_all_negative_no_positive():
    df = _make_valid_df()
    df["label"] = 0  # no positives
    with pytest.raises(ValueError, match="no positive pairs"):
        validate_pairs_df(df)


# ── validate_features_coverage ───────────────────────────────────────────────

def test_features_coverage_all_present():
    df = _make_valid_df()
    features = {
        "lfw_record_Alice_000001": np.zeros(32),
        "lfw_record_Alice_000002": np.zeros(32),
        "lfw_record_Bob_000010":   np.zeros(32),
        "lfw_record_Carol_000020": np.zeros(32),
    }
    validate_features_coverage(features, df)  # should not raise


def test_features_coverage_missing_id():
    df = _make_valid_df()
    features = {
        "lfw_record_Alice_000001": np.zeros(32),
        # missing Alice_000002, Bob_000010, Carol_000020
    }
    with pytest.raises(ValueError, match="not found in features"):
        validate_features_coverage(features, df)


# ── validate_metrics ─────────────────────────────────────────────────────────

def test_valid_metrics_pass():
    m = {"accuracy": 0.75, "FAR": 0.1, "FRR": 0.2, "balanced_accuracy": 0.8, "F1": 0.7}
    validate_metrics(m)  # should not raise


def test_metric_out_of_range():
    m = {"accuracy": 1.5}  # > 1
    with pytest.raises(ValueError, match="out of range"):
        validate_metrics(m)


def test_metric_negative():
    m = {"FAR": -0.1}
    with pytest.raises(ValueError, match="out of range"):
        validate_metrics(m)
