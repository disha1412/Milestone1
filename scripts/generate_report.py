"""
Generate the Milestone 2 evaluation report as a 2-page PDF.
Requires all 5 tracked runs to have been completed first.

Page 1: ROC curves (baseline vs improved) + Confusion matrices (baseline vs improved, test)
Page 2: Run summary table + Error slice analysis with hypotheses
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.evaluation import (
    compute_scores, threshold_sweep, select_threshold,
    evaluate_at_threshold, compute_auc, compute_error_slices,
)
from src.features import load_features
from src.tracking import RunTracker


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_pairs(path):
    return pd.read_csv(path)


def get_thresholds(config):
    tr = config["evaluation"]["threshold_range"]
    return np.linspace(tr["start"], tr["stop"], tr["steps"])


def build_roc_data(config, features, data_version):
    """Recompute sweep on val for the given data version."""
    if data_version == "baseline":
        pairs_dir = config["pairs"]["baseline_dir"]
    else:
        pairs_dir = config["pairs"]["improved_dir"]
    pairs_df = load_pairs(os.path.join(pairs_dir, "val_pairs.csv"))
    scores, labels = compute_scores(pairs_df, features)
    thresholds = get_thresholds(config)
    sweep_results = threshold_sweep(scores, labels, thresholds)
    auc = compute_auc(sweep_results)
    return sweep_results, auc


def build_cm_data(config, features, tracker, data_version):
    """Recompute metrics on test at the selected threshold for the given data version."""
    if data_version == "baseline":
        pairs_dir = config["pairs"]["baseline_dir"]
        threshold = tracker.get_selected_threshold("baseline_val_selected")
    else:
        pairs_dir = config["pairs"]["improved_dir"]
        threshold = tracker.get_selected_threshold("improved_test_final")
    pairs_df = load_pairs(os.path.join(pairs_dir, "test_pairs.csv"))
    scores, labels = compute_scores(pairs_df, features)
    metrics = evaluate_at_threshold(scores, labels, threshold)
    return metrics, threshold, scores, labels, pairs_df


def draw_roc_axes(ax, sweep_results, auc, color, label):
    fars = [r["FAR"] for r in sweep_results]
    tprs = [r["TPR"] for r in sweep_results]
    ax.plot(fars, tprs, color=color, linewidth=2, label=f"{label} (AUC={auc:.3f})")


def draw_cm_axes(ax, tp, tn, fp, fn, title):
    cm = np.array([[tn, fp], [fn, tp]])
    cell_labels = [
        ["TN\n(Diff→Diff)", "FP\n(Diff→Same)"],
        ["FN\n(Same→Diff)", "TP\n(Same→Same)"],
    ]
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cell_labels[i][j]}\n{cm[i,j]}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred\nDiff", "Pred\nSame"], fontsize=8)
    ax.set_yticklabels(["True\nDiff", "True\nSame"], fontsize=8)
    ax.set_title(title, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description="Generate Milestone 2 PDF report")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    tracker = RunTracker(config["tracking"]["runs_file"])
    report_path = config["report"]["output_path"]
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    print("[report] Loading features ...")
    features = load_features(config["features"]["output_path"])

    print("[report] Computing ROC data ...")
    base_sweep, base_auc = build_roc_data(config, features, "baseline")
    impr_sweep, impr_auc = build_roc_data(config, features, "improved")

    print("[report] Computing confusion matrix data ...")
    base_metrics, base_thresh, base_scores, base_labels, base_df = \
        build_cm_data(config, features, tracker, "baseline")
    impr_metrics, impr_thresh, impr_scores, impr_labels, impr_df = \
        build_cm_data(config, features, tracker, "improved")

    # Error slices for baseline test
    slices = compute_error_slices(base_df, base_scores, base_labels, base_thresh, features)

    runs = tracker.list_runs()

    with PdfPages(report_path) as pdf:

        # ── PAGE 1 ─────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("MSML 605 — Milestone 2 Evaluation Report", fontsize=13, fontweight="bold", y=0.98)

        # ROC curves (top row)
        ax_roc = fig.add_axes([0.08, 0.62, 0.40, 0.28])
        draw_roc_axes(ax_roc, base_sweep, base_auc, "steelblue", "Baseline")
        draw_roc_axes(ax_roc, impr_sweep, impr_auc, "darkorange", "Improved")
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random")
        ax_roc.set_xlabel("False Accept Rate (FAR)", fontsize=9)
        ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=9)
        ax_roc.set_title("ROC Curves — Validation Set", fontsize=10)
        ax_roc.legend(fontsize=8, loc="lower right")
        ax_roc.grid(True, alpha=0.3)
        ax_roc.set_xlim([0, 1])
        ax_roc.set_ylim([0, 1])

        # Threshold selection explanation (top right)
        ax_text = fig.add_axes([0.55, 0.62, 0.40, 0.28])
        ax_text.axis("off")
        info = [
            "Threshold Selection",
            f"Rule: maximize balanced accuracy on val set.",
            "",
            f"Baseline selected threshold:  {base_thresh:.4f}",
            f"  Balanced Acc (val):          {base_metrics.get('balanced_accuracy', 0):.4f}",
            f"  FAR (test):                  {base_metrics['FAR']:.4f}",
            f"  FRR (test):                  {base_metrics['FRR']:.4f}",
            "",
            f"Improved selected threshold:  {impr_thresh:.4f}",
            f"  Balanced Acc (val):          {impr_metrics.get('balanced_accuracy', 0):.4f}",
            f"  FAR (test):                  {impr_metrics['FAR']:.4f}",
            f"  FRR (test):                  {impr_metrics['FRR']:.4f}",
        ]
        ax_text.text(0.02, 0.95, "\n".join(info), transform=ax_text.transAxes,
                     fontsize=8, verticalalignment="top", fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        # Confusion matrices (middle row)
        ax_cm1 = fig.add_axes([0.08, 0.30, 0.35, 0.25])
        draw_cm_axes(ax_cm1, base_metrics["TP"], base_metrics["TN"],
                     base_metrics["FP"], base_metrics["FN"],
                     f"Baseline — Test (t={base_thresh:.3f})")

        ax_cm2 = fig.add_axes([0.55, 0.30, 0.35, 0.25])
        draw_cm_axes(ax_cm2, impr_metrics["TP"], impr_metrics["TN"],
                     impr_metrics["FP"], impr_metrics["FN"],
                     f"Improved — Test (t={impr_thresh:.3f})")

        # Score distribution (bottom)
        ax_dist = fig.add_axes([0.08, 0.06, 0.84, 0.18])
        pos_scores = base_scores[base_labels == 1]
        neg_scores = base_scores[base_labels == 0]
        bins = np.linspace(0, 1, 50)
        ax_dist.hist(neg_scores, bins=bins, alpha=0.6, color="salmon", label="Different (label=0)", density=True)
        ax_dist.hist(pos_scores, bins=bins, alpha=0.6, color="steelblue", label="Same (label=1)", density=True)
        ax_dist.axvline(base_thresh, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={base_thresh:.3f}")
        ax_dist.set_xlabel("Cosine Similarity Score", fontsize=9)
        ax_dist.set_ylabel("Density", fontsize=9)
        ax_dist.set_title("Baseline Score Distribution — Test Set", fontsize=10)
        ax_dist.legend(fontsize=8)
        ax_dist.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── PAGE 2 ─────────────────────────────────────────────────────────
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle("Run Summary & Error Analysis", fontsize=13, fontweight="bold", y=0.98)

        # Run summary table
        ax_table = fig2.add_axes([0.03, 0.72, 0.94, 0.22])
        ax_table.axis("off")

        table_data = []
        col_labels = ["Run ID", "Split", "Data", "Threshold", "BalAcc", "FAR", "FRR", "AUC", "Note"]
        for r in runs:
            thresh = r["threshold"]
            thresh_s = f"{thresh:.4f}" if isinstance(thresh, float) else str(thresh)
            m = r["metrics"]
            row = [
                r["run_id"][:28],
                r["split"],
                r["data_version"],
                thresh_s,
                f"{m.get('balanced_accuracy', float('nan')):.3f}" if isinstance(m.get('balanced_accuracy'), float) else "—",
                f"{m.get('FAR', float('nan')):.3f}" if isinstance(m.get('FAR'), float) else "—",
                f"{m.get('FRR', float('nan')):.3f}" if isinstance(m.get('FRR'), float) else "—",
                f"{m.get('auc', float('nan')):.3f}" if isinstance(m.get('auc'), float) else "—",
                r.get("note", "")[:40],
            ]
            table_data.append(row)

        tbl = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.5)
        tbl.scale(1, 1.4)
        ax_table.set_title("All 5 Tracked Runs", fontsize=11, pad=12)

        # Error slice analysis
        ax_slices = fig2.add_axes([0.03, 0.04, 0.94, 0.64])
        ax_slices.axis("off")

        s1 = slices.get("slice_rare_positives", {})
        s2 = slices.get("slice_hard_negatives", {})

        slice_text = [
            "ERROR ANALYSIS — Baseline (Test Set)",
            "=" * 70,
            "",
            "SLICE 1: Positive pairs from rare identities (≤3 images in dataset)",
            f"  Pairs in slice:     {s1.get('n_pairs', 'N/A')}",
            f"  False Reject Rate:  {s1.get('FRR', 'N/A')}",
            f"  False rejects:      {s1.get('false_reject_count', 'N/A')}",
            f"  Example missed IDs: {str(s1.get('example_missed_pairs', []))[:80]}",
            "",
            "  Hypothesis:",
        ] + [f"    {line}" for line in (s1.get("hypothesis", "") or "").split(". ") if line] + [
            "",
            "SLICE 2: Hard negative pairs (top 25% similarity among negatives)",
            f"  Pairs in slice:        {s2.get('n_pairs', 'N/A')}",
            f"  Similarity cutoff:     {s2.get('similarity_cutoff', 'N/A')}",
            f"  False Accept Rate:     {s2.get('FAR', 'N/A')}",
            f"  False accepts:         {s2.get('false_accept_count', 'N/A')}",
            f"  Example false-accept:  {str(s2.get('example_false_accept_pairs', []))[:80]}",
            "",
            "  Hypothesis:",
        ] + [f"    {line}" for line in (s2.get("hypothesis", "") or "").split(". ") if line] + [
            "",
            "DATA-CENTRIC CHANGE SUMMARY",
            "=" * 70,
            "  Improvement: Cap each identity to max 5 images during pair generation.",
            "  Motivation: Without capping, high-frequency identities (e.g. ~530 images)",
            "              dominate the positive pair pool, making evaluation unrepresentative.",
            "  Effect: Positive pairs are drawn more uniformly across identities,",
            "          producing a harder and more realistic evaluation benchmark.",
        ]

        ax_slices.text(
            0.02, 0.98, "\n".join(slice_text),
            transform=ax_slices.transAxes,
            fontsize=7.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
        )
        ax_slices.set_title("Error Slice Analysis", fontsize=11, pad=10)

        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    print(f"[report] PDF report saved to {report_path}")


if __name__ == "__main__":
    main()
