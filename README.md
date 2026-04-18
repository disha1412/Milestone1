# Face Verification Pipeline — Milestone 3

## Project Overview

This project builds a reproducible face verification system on the LFW dataset. Given two face images, the pipeline produces a cosine-similarity score between their face embeddings and a same/different-person binary decision with calibrated confidence.

Milestone 3 upgrades the representation from flattened pixel vectors (Milestone 2 baseline) to a proper face embedding stage using FaceNet (InceptionResnetV1 pretrained on VGGFace2). The verified system is packaged in Docker, exposed through a CLI, and characterized under concurrent load.

---

## What Changed in Milestone 3

| Area | Change |
|---|---|
| Representation | Replaced pixel cosine baseline with FaceNet InceptionResnetV1 embeddings (512-dim, VGGFace2) |
| Inference interface | New `scripts/cli_infer.py` with `pair` and `batch` subcommands |
| Threshold | Re-selected on val split using the same max-balanced-accuracy rule; locked to `0.35` in `configs/m3.yaml` |
| Confidence | Sigmoid-scaled margin: `sigmoid(10 * (score - threshold))`, range [0, 1] |
| Docker | `Dockerfile` added; full pipeline runs from a clean clone |
| Load test | `scripts/load_test.py` — concurrent inference with configurable workers, reports throughput and p95/p99 latency |
| New modules | `src/embeddings.py`, `src/inference.py` |
| Tests | Inference unit tests and four inference smoke/integration tests added |

---

## Repository Structure

```
repo_root/
├── Dockerfile
├── requirements.txt
├── configs/
│   ├── m1.yaml              # Milestone 1 config
│   ├── m2.yaml              # Milestone 2 baseline config
│   ├── m2_capped.yaml       # Milestone 2 data-centric v2 config
│   └── m3.yaml              # Milestone 3 config (embedding + inference + load test)
├── src/
│   ├── embeddings.py        # Preprocessing and FaceNet embedding extraction
│   ├── inference.py         # Pair-level inference: score, decision, confidence, latency
│   ├── similarity.py        # Vectorized cosine and Euclidean distance
│   ├── scoring.py           # Batch pair scorer using embeddings
│   ├── metrics.py           # ROC, confusion matrix, threshold selection
│   ├── validation.py        # Input/output validation checks
│   ├── tracking.py          # Run logging (JSONL + CSV)
│   └── error_analysis.py    # Error slicing utilities
├── scripts/
│   ├── cli_infer.py         # CLI inference interface (pair and batch modes)
│   ├── load_test.py         # Concurrency / load test script
│   ├── evaluate.py          # Threshold sweep, metrics, run logging
│   ├── ingest_lfw.py        # LFW ingestion and identity split
│   ├── make_pairs.py        # Deterministic pair generation
│   └── bench_similarity.py  # Vectorized vs loop benchmark
├── tests/
│   ├── conftest.py
│   ├── test_unit.py         # Unit tests (metrics, validation, inference, embeddings)
│   └── test_integration.py  # Full eval pipeline + inference smoke tests
├── reports/
│   └── MSML_MSAI 605 — Milestone 2 Report.pdf
└── outputs/                 # Generated artifacts (not committed)
    ├── runs/                # runs.jsonl + runs_summary.csv
    ├── eval/                # Sweep, ROC, confusion matrix, error analysis
    ├── load_test/           # Load test results JSON
    └── pairs/               # Generated pair CSVs
```

---

## Pipeline Summary

```
Image A, Image B
     │
     ▼
Preprocessing (resize to 160×160, per-image mean/std normalization)
     │
     ▼
Embedding extraction (FaceNet InceptionResnetV1, 512-dim)
     │
     ▼
Cosine similarity score  ∈ [-1, 1]
     │
     ▼
Threshold decision (score >= threshold → SAME, else DIFFERENT)
     │
     ▼
Calibrated confidence: sigmoid(10 × (score − threshold))  ∈ [0, 1]
```

**Embedding model:** FaceNet InceptionResnetV1 pretrained on VGGFace2 via `facenet-pytorch`. If unavailable, the pipeline falls back to a TF Hub MobileNetV2 feature vector, and finally to a normalized pixel vector.

**Threshold:** `0.35`, selected on the val split by maximizing balanced accuracy. Stored in `configs/m3.yaml` under `inference.threshold`. Do not tune on the test split.

**Confidence interpretation:** `0.5` means the score sits exactly on the threshold boundary. Values above `0.5` indicate increasing same-person confidence; values below `0.5` indicate increasing different-person confidence.

---

## How to Run

### Option A — Local environment

#### 1. Install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Ingest LFW and generate splits
```bash
python scripts/ingest_lfw.py --config configs/m3.yaml
```

#### 3. Generate pairs
```bash
python scripts/make_pairs.py --config configs/m3.yaml --data-version v1
```

#### 4. Re-select threshold on val (optional — `0.35` is already locked in config)
```bash
python scripts/evaluate.py --config configs/m3.yaml --split val --sweep --data-version v1
```

#### 5. CLI inference — single pair
```bash
python make_test_imgs.py
python scripts/cli_infer.py --config configs/m3.yaml pair --left test_a.jpg --right test_b.jpg
```

Expected output:
```
Pair:       pair_0
Left:       path/to/image_a.jpg
Right:      path/to/image_b.jpg
Score:      0.412381
Threshold:  0.350000
Decision:   SAME
Confidence: 0.815320
Latency:    42.17 ms
```

#### 6. CLI inference — batch mode
```bash
# 1. Ingest LFW
python scripts/ingest_lfw.py --config configs/m3.yaml

# 2. Generate pairs
python scripts/make_pairs.py --config configs/m3.yaml --data-version v1

# 3. Now batch inference will work
python scripts/cli_infer.py --config configs/m3.yaml batch `
    --pairs-file outputs/pairs/val_pairs.csv `
    --output outputs/eval/batch_results.json
```

#### 7. Run load test
```bash
python scripts/load_test.py `
    --config configs/m3.yaml `
    --pairs-file outputs/pairs/val_pairs.csv `
    --n-requests 50 `
    --n-workers 4
```

#### 8. Run tests
```bash
pytest tests/ -v
```

---

### Option B — Docker

#### Build
```bash
docker build -t face-verifier:v0.3 .
```

#### Single pair inference
```bash
docker run --rm `
    -v $(pwd)/configs:/app/configs `
    -v $(pwd):/app/hostfiles `
    face-verifier:v0.3 `
    --config configs/m3.yaml pair `
    --left /app/hostfiles/test_a.jpg `
    --right /app/hostfiles/test_b.jpg
```

#### Batch inference
```bash
docker run --rm `
    -v $(pwd)/configs:/app/configs `
    -v $(pwd)/outputs:/app/outputs `
    face-verifier:v0.3 `
    --config configs/m3.yaml batch `
    --pairs-file outputs/pairs/val_pairs.csv `
    --output outputs/eval/batch_results.json
```

#### Run tests inside Docker
```bash
docker run --rm --entrypoint pytest face-verifier:v0.3 tests/ -v
```

---

## Artifact Locations

| Artifact | Path |
|---|---|
| Tracked runs (JSONL) | `outputs/runs/runs.jsonl` |
| Run summary (CSV) | `outputs/runs/runs_summary.csv` |
| Val sweep results | `outputs/eval/val/sweep.json` |
| ROC curve | `outputs/eval/val/roc.png` |
| Selected threshold | `outputs/eval/val/selected_threshold.json` |
| Confusion matrix | `outputs/eval/val/confusion_matrix.png` |
| Error analysis | `outputs/eval/val/error_analysis.json` |
| Load test results | `outputs/load_test/results.json` |
| Batch inference output | `outputs/eval/batch_results.json` (if run) |
| Milestone 2 report | `reports/MSML_MSAI 605 — Milestone 2 Report.pdf` |

---

## Design Notes

**Embedding source:** FaceNet InceptionResnetV1 (`facenet-pytorch`, pretrained on VGGFace2). This is a standard identity-focused face embedding model, widely used as a classical face-verification baseline. Embedding dimensionality is 512.

**Threshold selection:** Same rule as Milestone 2 — sweep 100 evenly spaced thresholds on the val split, select the one maximizing balanced accuracy. The locked value `0.35` is stored under `inference.threshold` in `configs/m3.yaml` and is what the CLI and load test use.

**Confidence formula:** `sigmoid(10 × (score − threshold))`. The steepness factor of 10 means a score 0.2 above threshold maps to ~0.88 confidence, and 0.2 below maps to ~0.12. Output range is [0, 1]; 0.5 is the exact decision boundary.

**Stage separation:** `src/inference.py` times and returns preprocessing, embedding, and scoring latencies separately to support Milestone 4 profiling.

**Load test:** Uses `ProcessPoolExecutor` so workers are separate processes (avoids Python GIL). Pairs are drawn cyclically from the provided CSV so the test is deterministic for a given input file and `--n-requests` value.

---

## .gitignore Notes

Ignored: `data/`, `outputs/`, virtual environments, `__pycache__`, `.ipynb_checkpoints`, OS files, Docker build artifacts.

Committed: code, configs, tests, `Dockerfile`, `requirements.txt`, `reports/`.