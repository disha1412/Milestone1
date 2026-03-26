# MSML 605 — Face Verification Pipeline (Milestones 1 & 2)

## Project Overview

This project builds a reproducible face verification system on the LFW (Labeled Faces in the Wild)
dataset. Given two face images, the system produces a cosine similarity score and a same-person /
different-person decision based on a calibrated threshold.

**Milestone 1** established the deterministic backbone: LFW ingestion, identity-split pair
generation, and vectorized similarity scoring with benchmarking.

**Milestone 2** adds the disciplined evaluation loop: threshold calibration on a held-out
validation set, error analysis, a data-centric improvement (identity-count capping), experiment
tracking across 5 logged runs, and a 2-page PDF report.

---

## Repo Layout

```
configs/
  m1.yaml                  # seed, paths, split/pair/benchmark settings
  m2.yaml                  # feature extraction, evaluation, tracking, report paths
src/
  similarity.py            # vectorized cosine similarity and Euclidean distance (M1)
  features.py              # image feature extraction (32x32 grayscale → L2-norm vector)
  evaluation.py            # threshold sweep, metrics, ROC plot, confusion matrix
  tracking.py              # JSON-based run tracker
  validation.py            # pipeline input/output validation checks
scripts/
  ingest_lfw.py            # download LFW, split by identity, write manifest
  make_pairs.py            # generate deterministic baseline pair CSVs
  bench_similarity.py      # NumPy vs loop benchmark
  extract_features.py      # build outputs/features.npz from TFDS cache
  make_pairs_improved.py   # data-centric improvement: cap images per identity
  run_evaluation.py        # execute one of 5 tracked evaluation runs
  generate_report.py       # produce 2-page PDF report
tests/
  test_metrics.py          # unit tests for metric computation
  test_validation.py       # unit tests for validation checks
  test_integration.py      # end-to-end integration test (synthetic data, no TFDS)
reports/
  milestone2_report.pdf    # generated evaluation report (committed as grading evidence)
data/                      # TFDS dataset cache — NOT committed
outputs/                   # generated artifacts — NOT committed
```

---

## How to Run

### 1. Set up environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Milestone 1 — Ingest and prepare data

```bash
python scripts/ingest_lfw.py --config configs/m1.yaml
python scripts/make_pairs.py --config configs/m1.yaml
python scripts/bench_similarity.py --config configs/m1.yaml
```

### 3. Milestone 2 — Feature extraction and improved pairs

```bash
# Extract 32x32 grayscale L2-normalized features for all LFW images
python scripts/extract_features.py --config configs/m2.yaml

# Generate improved pairs (cap each identity to 5 images)
python scripts/make_pairs_improved.py --config configs/m2.yaml
```

### 4. Milestone 2 — 5 tracked evaluation runs (run in this order)

```bash
python scripts/run_evaluation.py --config configs/m2.yaml --run baseline_val_sweep
python scripts/run_evaluation.py --config configs/m2.yaml --run baseline_val_selected
python scripts/run_evaluation.py --config configs/m2.yaml --run baseline_test_final
python scripts/run_evaluation.py --config configs/m2.yaml --run improved_val_sweep
python scripts/run_evaluation.py --config configs/m2.yaml --run improved_test_final
```

Run results are logged to `outputs/runs.json`.

### 5. Generate PDF report

```bash
python scripts/generate_report.py --config configs/m2.yaml
# Output: reports/milestone2_report.pdf
```

### 6. Run tests

```bash
python -m pytest tests/ -v
```

---

## Outputs Summary

| File | Description |
|---|---|
| `outputs/manifest.json` | Seed, split policy, counts per split |
| `outputs/splits.json` | Identity IDs per split |
| `outputs/pairs/{split}_pairs.csv` | Baseline pairs: left_path, right_path, label |
| `outputs/pairs_improved/{split}_pairs.csv` | Improved pairs (identity-capped) |
| `outputs/features.npz` | Feature vectors for all LFW images |
| `outputs/eval/baseline_val_sweep.json` | Threshold sweep results (baseline) |
| `outputs/eval/improved_val_sweep.json` | Threshold sweep results (improved) |
| `outputs/eval/plots/*.png` | ROC curves and confusion matrices |
| `outputs/runs.json` | All 5 tracked run records |
| `reports/milestone2_report.pdf` | 2-page evaluation report |

---

## Threshold Selection Rule

Threshold is selected on the **validation set** by maximizing balanced accuracy
`(TPR + TNR) / 2`. The same rule is applied for both baseline and improved-data runs
before any test-set inspection.

---

## Data-Centric Improvement

**Problem:** Without capping, high-frequency identities dominate the positive pair pool,
making evaluation unrepresentative.

**Change:** Cap each identity to at most 5 images before pair sampling
(`improved_max_images_per_identity: 5` in `m2.yaml`). This produces a more diverse and
uniformly difficult positive evaluation set.

---

## Determinism Notes

- Seed: `42` in both configs
- All shuffles use `random.Random(seed)` with sorted inputs
- Feature extraction iterates TFDS with `shuffle_files=False`
- Running any script twice produces identical outputs

---

## Git Tags

- `v0.1` — Milestone 1
- `v0.2` — Milestone 2
