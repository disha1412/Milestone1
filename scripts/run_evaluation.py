"""
Run one of the 5 tracked evaluation runs for Milestone 2.

Usage:
  python scripts/run_evaluation.py --config configs/m2.yaml --run baseline_val_sweep
  python scripts/run_evaluation.py --config configs/m2.yaml --run baseline_val_selected
  python scripts/run_evaluation.py --config configs/m2.yaml --run baseline_test_final
  python scripts/run_evaluation.py --config configs/m2.yaml --run improved_val_sweep
  python scripts/run_evaluation.py --config configs/m2.yaml --run improved_test_final

Run order: sweep must come before selected/final for each data version.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

from src.evaluation import (
    compute_scores,
    threshold_sweep,
    select_threshold,
    evaluate_at_threshold,
    compute_auc,
    compute_error_slices,
    plot_roc,
    plot_confusion_matrix,
)
from src.features import load_features
from src.tracking import RunTracker
from src.validation import validate_pairs_df, validate_features_coverage, validate_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_pairs(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_thresholds(config: dict) -> np.ndarray:
    tr = config["evaluation"]["threshold_range"]
    return np.linspace(tr["start"], tr["stop"], tr["steps"])


# ── Run 1: Baseline threshold sweep on validation ────────────────────────────
def run_baseline_val_sweep(config, features, tracker):
    pairs_dir = config["pairs"]["baseline_dir"]
    pairs_df = load_pairs(os.path.join(pairs_dir, "val_pairs.csv"))
    validate_pairs_df(pairs_df)
    validate_features_coverage(features, pairs_df)

    scores, labels = compute_scores(pairs_df, features)
    thresholds = get_thresholds(config)
    sweep_results = threshold_sweep(scores, labels, thresholds)
    auc = compute_auc(sweep_results)

    # Save sweep results for next run to load
    eval_dir = config["evaluation"]["output_dir"]
    os.makedirs(eval_dir, exist_ok=True)
    sweep_path = os.path.join(eval_dir, "baseline_val_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f)

    # ROC plot
    plot_path = os.path.join(eval_dir, "plots", "baseline_val_roc.png")
    plot_roc(sweep_results, plot_path, title="Baseline ROC — Validation")

    tracker.log_run(
        run_id="baseline_val_sweep",
        config_name="m2.yaml",
        split="val",
        data_version="baseline",
        threshold="sweep",
        metrics={"auc": round(auc, 4), "n_thresholds": len(thresholds)},
        artifacts=[sweep_path, plot_path],
        note="Baseline threshold sweep on validation set. AUC computed from sweep.",
    )
    print(f"[run] baseline_val_sweep complete. AUC = {auc:.4f}")


# ── Run 2: Baseline — select threshold on validation ─────────────────────────
def run_baseline_val_selected(config, features, tracker):
    eval_dir = config["evaluation"]["output_dir"]
    sweep_path = os.path.join(eval_dir, "baseline_val_sweep.json")
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(
            f"Sweep results not found at {sweep_path}. "
            "Run baseline_val_sweep first."
        )
    with open(sweep_path) as f:
        sweep_results = json.load(f)

    rule = config["evaluation"]["threshold_rule"]
    best_threshold, best_metrics = select_threshold(sweep_results, rule)
    validate_metrics(best_metrics)

    # Also evaluate at the selected threshold to get confusion matrix
    pairs_df = load_pairs(os.path.join(config["pairs"]["baseline_dir"], "val_pairs.csv"))
    scores, labels = compute_scores(pairs_df, features)
    metrics = evaluate_at_threshold(scores, labels, best_threshold)

    cm_path = os.path.join(eval_dir, "plots", "baseline_val_cm.png")
    plot_confusion_matrix(
        metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"],
        cm_path, title=f"Baseline Confusion Matrix — Val (t={best_threshold:.3f})",
    )

    logged_metrics = {
        k: round(float(v), 4) for k, v in metrics.items()
        if isinstance(v, (int, float))
    }
    tracker.log_run(
        run_id="baseline_val_selected",
        config_name="m2.yaml",
        split="val",
        data_version="baseline",
        threshold=round(best_threshold, 4),
        metrics=logged_metrics,
        artifacts=[cm_path],
        note=f"Selected threshold using rule '{rule}' on validation sweep results.",
    )
    print(f"[run] baseline_val_selected complete. Threshold={best_threshold:.4f}, "
          f"BalAcc={metrics['balanced_accuracy']:.4f}")


# ── Run 3: Baseline — final evaluation on test set ───────────────────────────
def run_baseline_test_final(config, features, tracker):
    selected_threshold = tracker.get_selected_threshold("baseline_val_selected")

    pairs_df = load_pairs(os.path.join(config["pairs"]["baseline_dir"], "test_pairs.csv"))
    validate_pairs_df(pairs_df)
    validate_features_coverage(features, pairs_df)

    scores, labels = compute_scores(pairs_df, features)
    metrics = evaluate_at_threshold(scores, labels, selected_threshold)

    # Error slice analysis
    slices = compute_error_slices(pairs_df, scores, labels, selected_threshold, features)

    eval_dir = config["evaluation"]["output_dir"]
    cm_path = os.path.join(eval_dir, "plots", "baseline_test_cm.png")
    plot_confusion_matrix(
        metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"],
        cm_path, title=f"Baseline Confusion Matrix — Test (t={selected_threshold:.3f})",
    )

    slices_path = os.path.join(eval_dir, "baseline_test_slices.json")
    with open(slices_path, "w") as f:
        json.dump(slices, f, indent=2)

    logged_metrics = {
        k: round(float(v), 4) for k, v in metrics.items()
        if isinstance(v, (int, float))
    }
    tracker.log_run(
        run_id="baseline_test_final",
        config_name="m2.yaml",
        split="test",
        data_version="baseline",
        threshold=round(selected_threshold, 4),
        metrics=logged_metrics,
        artifacts=[cm_path, slices_path],
        note="Baseline final evaluation on test set using threshold from val selection.",
    )
    print(f"[run] baseline_test_final complete. "
          f"Accuracy={metrics['accuracy']:.4f}, FAR={metrics['FAR']:.4f}, FRR={metrics['FRR']:.4f}")


# ── Run 4: Improved data — threshold sweep on validation ─────────────────────
def run_improved_val_sweep(config, features, tracker):
    improved_dir = config["pairs"]["improved_dir"]
    pairs_path = os.path.join(improved_dir, "val_pairs.csv")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(
            f"Improved pairs not found at {pairs_path}. "
            "Run make_pairs_improved.py first."
        )

    pairs_df = load_pairs(pairs_path)
    validate_pairs_df(pairs_df)
    validate_features_coverage(features, pairs_df)

    scores, labels = compute_scores(pairs_df, features)
    thresholds = get_thresholds(config)
    sweep_results = threshold_sweep(scores, labels, thresholds)
    auc = compute_auc(sweep_results)

    eval_dir = config["evaluation"]["output_dir"]
    os.makedirs(eval_dir, exist_ok=True)
    sweep_path = os.path.join(eval_dir, "improved_val_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f)

    plot_path = os.path.join(eval_dir, "plots", "improved_val_roc.png")
    plot_roc(sweep_results, plot_path, title="Improved Data ROC — Validation")

    tracker.log_run(
        run_id="improved_val_sweep",
        config_name="m2.yaml",
        split="val",
        data_version="improved",
        threshold="sweep",
        metrics={"auc": round(auc, 4), "n_thresholds": len(thresholds)},
        artifacts=[sweep_path, plot_path],
        note=f"Improved data (max {config['pairs']['improved_max_images_per_identity']} "
             f"images/identity) threshold sweep on validation.",
    )
    print(f"[run] improved_val_sweep complete. AUC = {auc:.4f}")


# ── Run 5: Improved data — final evaluation on test set ──────────────────────
def run_improved_test_final(config, features, tracker):
    # Select threshold from improved val sweep
    eval_dir = config["evaluation"]["output_dir"]
    sweep_path = os.path.join(eval_dir, "improved_val_sweep.json")
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(
            f"Improved sweep results not found at {sweep_path}. "
            "Run improved_val_sweep first."
        )
    with open(sweep_path) as f:
        sweep_results = json.load(f)

    rule = config["evaluation"]["threshold_rule"]
    best_threshold, _ = select_threshold(sweep_results, rule)

    pairs_df = load_pairs(os.path.join(config["pairs"]["improved_dir"], "test_pairs.csv"))
    validate_pairs_df(pairs_df)
    validate_features_coverage(features, pairs_df)

    scores, labels = compute_scores(pairs_df, features)
    metrics = evaluate_at_threshold(scores, labels, best_threshold)

    cm_path = os.path.join(eval_dir, "plots", "improved_test_cm.png")
    plot_confusion_matrix(
        metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"],
        cm_path, title=f"Improved Confusion Matrix — Test (t={best_threshold:.3f})",
    )

    logged_metrics = {
        k: round(float(v), 4) for k, v in metrics.items()
        if isinstance(v, (int, float))
    }
    tracker.log_run(
        run_id="improved_test_final",
        config_name="m2.yaml",
        split="test",
        data_version="improved",
        threshold=round(best_threshold, 4),
        metrics=logged_metrics,
        artifacts=[cm_path],
        note="Improved data final evaluation on test set using threshold from improved val sweep.",
    )
    print(f"[run] improved_test_final complete. "
          f"Accuracy={metrics['accuracy']:.4f}, FAR={metrics['FAR']:.4f}, FRR={metrics['FRR']:.4f}")


DISPATCH = {
    "baseline_val_sweep": run_baseline_val_sweep,
    "baseline_val_selected": run_baseline_val_selected,
    "baseline_test_final": run_baseline_test_final,
    "improved_val_sweep": run_improved_val_sweep,
    "improved_test_final": run_improved_test_final,
}


def main():
    parser = argparse.ArgumentParser(description="Run a tracked Milestone 2 evaluation")
    parser.add_argument("--config", required=True, help="Path to m2.yaml")
    parser.add_argument(
        "--run", required=True,
        choices=list(DISPATCH.keys()),
        help="Which run to execute",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    features_path = config["features"]["output_path"]
    if not os.path.exists(features_path):
        print(f"[run] Features file not found at {features_path}.")
        print("[run] Run: python scripts/extract_features.py --config configs/m2.yaml")
        sys.exit(1)

    print(f"[run] Loading features from {features_path} ...")
    features = load_features(features_path)
    tracker = RunTracker(config["tracking"]["runs_file"])

    print(f"[run] Starting run: {args.run}")
    DISPATCH[args.run](config, features, tracker)


if __name__ == "__main__":
    main()
