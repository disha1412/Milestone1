"""
Evaluation utilities for face verification.
Score direction: higher cosine similarity -> more likely same person.
Decision rule: predict same-person (1) when score >= threshold.
"""

import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple


def compute_scores(pairs_df: pd.DataFrame, features: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Look up feature vectors and compute cosine similarity for each pair.
    Returns (scores, labels) as 1D float32 arrays.
    """
    from src.similarity import cosine_similarity_vectorized

    left_vecs = np.stack([features[r] for r in pairs_df["left_path"]])
    right_vecs = np.stack([features[r] for r in pairs_df["right_path"]])
    scores = cosine_similarity_vectorized(left_vecs, right_vecs)
    labels = pairs_df["label"].to_numpy()
    return scores.astype(np.float32), labels.astype(int)


def _binary_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute standard binary classification metrics."""
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    n = len(labels)
    accuracy = (tp + tn) / n if n > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # specificity
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # false accept rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # false reject rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0

    return dict(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        TPR=tpr,
        TNR=tnr,
        FAR=far,
        FRR=frr,
        precision=precision,
        F1=f1,
        TP=tp, TN=tn, FP=fp, FN=fn,
    )


def threshold_sweep(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> List[Dict]:
    """
    Evaluate metrics across a range of thresholds.
    Returns a list of dicts, one per threshold.
    """
    results = []
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        m = _binary_metrics(preds, labels)
        m["threshold"] = float(thresh)
        results.append(m)
    return results


def compute_auc(sweep_results: List[Dict]) -> float:
    """Compute AUC from sweep results using trapezoidal integration."""
    fars = np.array([r["FAR"] for r in sweep_results])
    tprs = np.array([r["TPR"] for r in sweep_results])
    order = np.argsort(fars)
    return float(np.trapz(tprs[order], fars[order]))


def select_threshold(sweep_results: List[Dict], rule: str = "maximize_balanced_accuracy") -> Tuple[float, dict]:
    """
    Select an operating threshold from sweep results using a stated rule.
    rule: 'maximize_balanced_accuracy' (default)
    Returns (selected_threshold, metrics_at_threshold).
    """
    if rule == "maximize_balanced_accuracy":
        best = max(sweep_results, key=lambda r: r["balanced_accuracy"])
    else:
        raise ValueError(f"Unknown threshold rule: {rule!r}")
    return best["threshold"], {k: v for k, v in best.items() if k != "threshold"}


def evaluate_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict:
    """Evaluate at a single threshold. Returns metrics dict."""
    preds = (scores >= threshold).astype(int)
    return _binary_metrics(preds, labels)


def compute_error_slices(
    pairs_df: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    features: dict,
) -> Dict[str, dict]:
    """
    Analyze two error slices:
    - slice_rare_positives: positive pairs from identities with few images (<=3 records)
    - slice_hard_negatives: negative pairs with similarity in top 25% of all negatives

    Returns dict mapping slice name -> {n_pairs, FAR or FRR, errors, hypothesis}.
    """
    from collections import Counter

    preds = (scores >= threshold).astype(int)

    # Count images per identity (identity = record_id[11:-7])
    identity_counts = Counter(rid[11:-7] for rid in features.keys())

    left_ids = pairs_df["left_path"].tolist()

    # --- Slice 1: positive pairs from rare identities (<=3 images total) ---
    rare_mask = np.array([
        (labels[i] == 1) and (identity_counts.get(left_ids[i][11:-7], 0) <= 3)
        for i in range(len(labels))
    ])
    if rare_mask.sum() > 0:
        s1_labels = labels[rare_mask]
        s1_preds = preds[rare_mask]
        s1_fn = int(((s1_preds == 0) & (s1_labels == 1)).sum())
        s1_frr = s1_fn / rare_mask.sum()
        s1_error_pairs = [
            left_ids[i]
            for i in range(len(labels))
            if rare_mask[i] and preds[i] == 0 and labels[i] == 1
        ][:5]
        slice1 = {
            "n_pairs": int(rare_mask.sum()),
            "FRR": round(s1_frr, 4),
            "false_reject_count": s1_fn,
            "example_missed_pairs": s1_error_pairs,
            "hypothesis": (
                "Identities with very few images produce harder positive pairs. "
                "With only one or two possible positive combinations, variability in "
                "pose and lighting is uncontrolled, leading to lower similarity scores "
                "and elevated false reject rates."
            ),
        }
    else:
        slice1 = {"n_pairs": 0, "note": "No rare-identity positive pairs in this split."}

    # --- Slice 2: hard negatives (negative pairs with top-25% similarity scores) ---
    neg_mask = labels == 0
    if neg_mask.sum() > 0:
        neg_scores = scores[neg_mask]
        cutoff = np.percentile(neg_scores, 75)
        hard_neg_mask = neg_mask & (scores >= cutoff)
        s2_preds = preds[hard_neg_mask]
        s2_labels = labels[hard_neg_mask]
        s2_fp = int(((s2_preds == 1) & (s2_labels == 0)).sum())
        s2_far = s2_fp / hard_neg_mask.sum() if hard_neg_mask.sum() > 0 else 0.0
        s2_error_pairs = [
            left_ids[i]
            for i in range(len(labels))
            if hard_neg_mask[i] and preds[i] == 1 and labels[i] == 0
        ][:5]
        slice2 = {
            "n_pairs": int(hard_neg_mask.sum()),
            "similarity_cutoff": round(float(cutoff), 4),
            "FAR": round(s2_far, 4),
            "false_accept_count": s2_fp,
            "example_false_accept_pairs": s2_error_pairs,
            "hypothesis": (
                "The hardest negative pairs have high pixel-level similarity, likely "
                "due to similar background, lighting, or shared demographic features. "
                "Pixel-based cosine similarity cannot distinguish identity-irrelevant "
                "visual similarity, causing false accepts in this subset."
            ),
        }
    else:
        slice2 = {"n_pairs": 0, "note": "No negative pairs in this split."}

    return {"slice_rare_positives": slice1, "slice_hard_negatives": slice2}


def plot_roc(sweep_results: List[Dict], output_path: str, title: str = "ROC Curve"):
    """Plot FAR vs TPR curve and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fars = [r["FAR"] for r in sweep_results]
    tprs = [r["TPR"] for r in sweep_results]
    auc = compute_auc(sweep_results)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fars, tprs, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] ROC plot saved to {output_path}")


def plot_confusion_matrix(
    tp: int, tn: int, fp: int, fn: int,
    output_path: str,
    title: str = "Confusion Matrix",
):
    """Plot a 2x2 confusion matrix and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array([[tn, fp], [fn, tp]])
    labels = [["TN\n(Diff→Diff)", "FP\n(Diff→Same)"],
              ["FN\n(Same→Diff)", "TP\n(Same→Same)"]]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nDifferent", "Predicted\nSame"])
    ax.set_yticklabels(["True\nDifferent", "True\nSame"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Confusion matrix saved to {output_path}")
