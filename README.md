# Face Verification Pipeline — v1.0-final (Milestone 4)

## Project Overview

This project builds a reproducible face verification system on the LFW dataset.
Given two face images, the pipeline produces a cosine-similarity score between
their FaceNet embeddings and a same/different-person binary decision with
calibrated confidence.

| Milestone | Main Contribution |
|---|---|
| Milestone 1 | Deterministic ingestion, saved pairs, reproducible structure, vectorized scoring |
| Milestone 2 | Threshold calibration, tracked runs, error analysis, data-centric iteration, validation |
| Milestone 3 | FaceNet embedding inference, Docker packaging, CLI interface, load testing |
| **Milestone 4** | **System Card, profiling report, reproducibility checklist, final release alignment** |

**Final tag:** `v1.0-final`

---

## Pipeline Summary

```
Image A, Image B
     │
     ▼
Preprocessing (resize to 160×160, per-image mean/std normalization)
     │
     ▼
Embedding extraction (FaceNet InceptionResnetV1, 512-dim, VGGFace2)
     │
     ▼
Cosine similarity score  ∈ [-1, 1]
     │
     ▼
Threshold decision (score >= 0.35 → SAME, else DIFFERENT)
     │
     ▼
Calibrated confidence: sigmoid(10 × (score − 0.35))  ∈ [0, 1]
```

**Operating threshold:** `0.35` — selected on the val split by maximizing balanced accuracy.  
**Confidence:** `0.5` = exactly on the decision boundary. Above `0.5` = same-person lean.  
**Model fallback:** FaceNet → TF Hub MobileNetV2 → normalized pixel vector (threshold only valid for FaceNet).

---

## Final Deliverables (Milestone 4)

| Artifact | Location |
|---|---|
| **System Card** | `reports/system_card.md` / `reports/system_card.pdf` |
| **Profiling Report** | `reports/profiling_report.md` / `reports/profiling_report.pdf` |
| **Reproducibility Checklist** | `reports/reproducibility_checklist.md` |
| Final inference config | `configs/m3.yaml` |
| Milestone 2 report | `reports/MSML_MSAI 605 — Milestone 2 Report.pdf` |
| Profiling script | `scripts/profile_latency.py` |

---

## Repository Structure

```
repo_root/
├── Dockerfile
├── requirements.txt
├── make_test_imgs.py
├── configs/
│   ├── m1.yaml
│   ├── m2.yaml
│   ├── m2_capped.yaml
│   └── m3.yaml                  ← Final inference config (threshold = 0.35)
├── src/
│   ├── embeddings.py            ← FaceNet preprocessing + embedding extraction
│   ├── inference.py             ← Pair inference: score, decision, confidence, latency
│   ├── similarity.py            ← Vectorized cosine and Euclidean distance
│   ├── scoring.py               ← Batch pair scorer
│   ├── metrics.py               ← ROC, confusion matrix, threshold selection
│   ├── validation.py            ← Input/output validation checks
│   ├── tracking.py              ← Run logging (JSONL + CSV)
│   └── error_analysis.py        ← Error slicing utilities
├── scripts/
│   ├── cli_infer.py             ← CLI inference (pair and batch modes)
│   ├── profile_latency.py       ← [NEW] Hardware-aware profiling script
│   ├── load_test.py             ← Concurrency / load test
│   ├── evaluate.py              ← Threshold sweep, metrics, run logging
│   ├── ingest_lfw.py            ← LFW ingestion and identity split
│   ├── make_pairs.py            ← Deterministic pair generation
│   └── bench_similarity.py      ← Vectorized vs loop benchmark
├── tests/
│   ├── conftest.py
│   ├── test_unit.py
│   └── test_integration.py
├── reports/
│   ├── system_card.md           ← [NEW] Final System Card
│   ├── system_card.pdf          ← [NEW] System Card PDF
│   ├── profiling_report.md      ← [NEW] Profiling report
│   ├── profiling_report.pdf     ← [NEW] Profiling report PDF
│   ├── reproducibility_checklist.md  ← [NEW] Exact reproduction commands
│   └── MSML_MSAI 605 — Milestone 2 Report.pdf
└── outputs/                     ← Generated artifacts (not committed)
    ├── runs/                    ← runs.jsonl + runs_summary.csv
    ├── eval/                    ← Sweep, ROC, confusion matrix, error analysis
    ├── profiling/               ← profiling results JSON
    ├── load_test/               ← Load test results
    └── pairs/                   ← Generated pair CSVs
```

---

## How to Run

### Quick start (no LFW required)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic test images
python make_test_imgs.py

# 3. Run single-pair CLI inference
python scripts/cli_infer.py --config configs/m3.yaml pair \
  --left test_a.jpg \
  --right test_b.jpg

# 4. Run test suite
pytest tests/ -v

# 5. Run hardware profiling (CPU baseline)
python scripts/profile_latency.py \
  --config configs/m3.yaml \
  --n-repeats 30 \
  --output outputs/profiling/results.json
```

### Full pipeline with LFW data

```bash
# Ingest LFW
python scripts/ingest_lfw.py --config configs/m3.yaml

# Generate pairs
python scripts/make_pairs.py --config configs/m3.yaml --data-version v1

# Threshold sweep on val
python scripts/evaluate.py --config configs/m3.yaml --split val --sweep --data-version v1

# Final evaluation on test (locked threshold = 0.35)
python scripts/evaluate.py --config configs/m3.yaml --split test --threshold 0.35 --data-version v1

# Batch inference
python scripts/cli_infer.py --config configs/m3.yaml batch \
  --pairs-file outputs/pairs/val_pairs.csv \
  --output outputs/eval/batch_results.json

# Load test (50 requests, 4 workers)
python scripts/load_test.py \
  --config configs/m3.yaml \
  --pairs-file outputs/pairs/val_pairs.csv \
  --n-requests 50 \
  --n-workers 4
```

### Docker

```bash
# Build
docker build -t face-verifier:v1.0-final .

# Single pair inference
docker run --rm \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd):/app/hostfiles \
  face-verifier:v1.0-final \
  --config configs/m3.yaml pair \
  --left /app/hostfiles/test_a.jpg \
  --right /app/hostfiles/test_b.jpg

# Run tests inside Docker
docker run --rm --entrypoint pytest face-verifier:v1.0-final tests/ -v
```

---

## Key Configuration (configs/m3.yaml)

| Parameter | Value |
|---|---|
| Operating threshold | `0.35` |
| Threshold selection rule | `max_balanced_accuracy` |
| Threshold selected on | `val` split |
| Embedding model | FaceNet InceptionResnetV1 (VGGFace2) |
| Embedding dimension | 512 |
| Confidence formula | `sigmoid(10 × (score − threshold))` |
| Dataset | LFW via TensorFlow Datasets |
| Split policy | 70/15/15 by identity |

---

## Artifact Locations

| Artifact | Path |
|---|---|
| System Card | `reports/system_card.pdf` |
| Profiling Report | `reports/profiling_report.pdf` |
| Reproducibility Checklist | `reports/reproducibility_checklist.md` |
| Tracked runs (JSONL) | `outputs/runs/runs.jsonl` |
| Run summary (CSV) | `outputs/runs/runs_summary.csv` |
| Val sweep | `outputs/eval/val/sweep.json` |
| ROC curve | `outputs/eval/val/roc.png` |
| Selected threshold | `outputs/eval/val/selected_threshold.json` |
| Confusion matrix | `outputs/eval/val/confusion_matrix.png` |
| Error analysis | `outputs/eval/val/error_analysis.json` |
| Profiling results | `outputs/profiling/results.json` |
| Load test results | `outputs/load_test/results.json` |
| Batch inference output | `outputs/eval/batch_results.json` |
| Milestone 2 report | `reports/MSML_MSAI 605 — Milestone 2 Report.pdf` |

---

## Design Notes

**Embedding:** FaceNet InceptionResnetV1 pretrained on VGGFace2 via `facenet-pytorch`.
Embedding dimension is 512. If `facenet-pytorch` is unavailable, the pipeline
falls back to TF Hub MobileNetV2 features, and finally to a normalized pixel vector.
The threshold of 0.35 is only calibrated for the FaceNet backend.

**Threshold:** Selected on the val split by sweeping 100 candidate thresholds
and maximizing balanced accuracy. Stored in `configs/m3.yaml` under
`inference.threshold`. Do not re-tune on the test split.

**Confidence:** `sigmoid(10 × (score − threshold))`. Steepness factor 10 means
a score 0.2 above threshold ≈ 0.88 confidence; 0.2 below ≈ 0.12 confidence.

**Profiling:** Stage-level latency (preprocessing, embedding, scoring) is
reported separately to support runtime analysis. The embedding stage dominates.

**Load test:** Uses `ProcessPoolExecutor` to avoid Python GIL. Pairs are drawn
cyclically from the CSV for deterministic test behavior.

**Responsible use:** See the System Card (`reports/system_card.pdf`) for a full
discussion of intended use, limitations, failure modes, and fairness-related risks.

---

## .gitignore Notes

Ignored: `data/`, `outputs/`, virtual environments, `__pycache__`, OS files,
notebook checkpoints, Docker build artifacts.

Committed: all source code, configs, tests, `Dockerfile`, `requirements.txt`,
`reports/`, profiling scripts, and the reproducibility checklist.
