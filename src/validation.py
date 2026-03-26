"""
Pipeline validation checks.
Validates pair DataFrames and feature map coverage before evaluation.
Fails early with descriptive errors to prevent silent failures.
"""

import pandas as pd
import numpy as np


REQUIRED_PAIR_COLUMNS = {"left_path", "right_path", "label"}
VALID_LABELS = {0, 1}


def validate_pairs_df(df: pd.DataFrame) -> None:
    """
    Validate a pairs DataFrame.
    Raises ValueError if any check fails.
    """
    missing_cols = REQUIRED_PAIR_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Pairs DataFrame missing columns: {missing_cols}")

    if df.empty:
        raise ValueError("Pairs DataFrame is empty.")

    null_counts = df[["left_path", "right_path", "label"]].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Pairs DataFrame has null values: {null_counts[null_counts > 0].to_dict()}")

    bad_labels = set(df["label"].unique()) - VALID_LABELS
    if bad_labels:
        raise ValueError(f"Invalid label values found: {bad_labels}. Expected values in {{0, 1}}.")

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    if n_pos == 0:
        raise ValueError("Pairs DataFrame has no positive pairs (label=1).")
    if n_neg == 0:
        raise ValueError("Pairs DataFrame has no negative pairs (label=0).")


def validate_features_coverage(features: dict, pairs_df: pd.DataFrame) -> None:
    """
    Check that every record ID in the pairs DataFrame exists in the features dict.
    Raises ValueError listing missing IDs if any are absent.
    """
    all_ids = set(pairs_df["left_path"]) | set(pairs_df["right_path"])
    missing = all_ids - set(features.keys())
    if missing:
        examples = list(missing)[:5]
        raise ValueError(
            f"{len(missing)} record IDs in pairs not found in features dict. "
            f"Examples: {examples}"
        )


def validate_metrics(metrics: dict) -> None:
    """
    Validate that computed metrics are within expected ranges.
    Raises ValueError if anything is out of range.
    """
    bounded = ["accuracy", "balanced_accuracy", "FAR", "FRR", "TPR", "TNR", "F1"]
    for key in bounded:
        if key in metrics:
            val = metrics[key]
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Metric '{key}' out of range [0, 1]: {val}")
