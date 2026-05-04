# Reproducibility Checklist — Face Verification Pipeline (v1.0-final)

This checklist provides the exact commands to reproduce the core results and run
the Dockerized CLI from a clean clone. Follow steps in order.

---

## 0. Prerequisites

- Python 3.10+
- Docker (for Option B)
- Git

---

## 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd <repo-root>

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Generate test images (no LFW required)

```bash
python make_test_imgs.py
# Produces: test_a.jpg, test_b.jpg
```

---

## 3. Run the CLI on a sample pair (Option A — local)

```bash
python scripts/cli_infer.py --config configs/m3.yaml pair \
  --left test_a.jpg \
  --right test_b.jpg
```

**Expected output structure:**
```
Pair:       pair_0
Left:       test_a.jpg
Right:      test_b.jpg
Score:      <float in [-1, 1]>
Threshold:  0.350000
Decision:   SAME or DIFFERENT
Confidence: <float in [0, 1]>
Latency:    <float> ms
```

---

## 4. Run the test suite

```bash
pytest tests/ -v
```

All tests should pass. The integration and unit tests do not require LFW data.

---

## 5. Run hardware-aware profiling (CPU baseline)

```bash
python scripts/profile_latency.py \
  --config configs/m3.yaml \
  --n-repeats 30 \
  --output outputs/profiling/results.json
```

Output is saved to `outputs/profiling/results.json`.
The profiling report PDF is in `reports/profiling_report.pdf`.

---

## 6. (Optional) Full pipeline with LFW data

If you have access to LFW via TensorFlow Datasets:

```bash
# Step 6a: Ingest LFW
python scripts/ingest_lfw.py --config configs/m3.yaml

# Step 6b: Generate pairs
python scripts/make_pairs.py --config configs/m3.yaml --data-version v1

# Step 6c: Threshold sweep on val
python scripts/evaluate.py --config configs/m3.yaml --split val --sweep --data-version v1

# Step 6d: Final test evaluation (use locked threshold 0.35)
python scripts/evaluate.py --config configs/m3.yaml --split test --threshold 0.35 --data-version v1

# Step 6e: Batch inference
python scripts/cli_infer.py --config configs/m3.yaml batch \
  --pairs-file outputs/pairs/val_pairs.csv \
  --output outputs/eval/batch_results.json

# Step 6f: Load test
python scripts/load_test.py \
  --config configs/m3.yaml \
  --pairs-file outputs/pairs/val_pairs.csv \
  --n-requests 50 \
  --n-workers 4
```

---

## 7. Option B — Docker

```bash
# Build the image
docker build -t face-verifier:v1.0-final .

# Run single-pair inference
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

## 8. Key artifact locations

| Artifact | Path |
|---|---|
| System Card | `reports/system_card.pdf` |
| Profiling Report | `reports/profiling_report.pdf` |
| Reproducibility Checklist | `reports/reproducibility_checklist.md` (this file) |
| Final config (inference) | `configs/m3.yaml` |
| Tracked runs (JSONL) | `outputs/runs/runs.jsonl` |
| Run summary (CSV) | `outputs/runs/runs_summary.csv` |
| Profiling results (JSON) | `outputs/profiling/results.json` |
| Val sweep | `outputs/eval/val/sweep.json` |
| ROC curve | `outputs/eval/val/roc.png` |
| Selected threshold | `outputs/eval/val/selected_threshold.json` |
| Milestone 2 report | `reports/MSML_MSAI 605 — Milestone 2 Report.pdf` |

---

## 9. Final Git tag

```bash
git tag v1.0-final
git push origin v1.0-final
```

The final release tag is: **v1.0-final**

---

## 10. Key configuration values (for reference)

| Parameter | Value | Source |
|---|---|---|
| Operating threshold | `0.35` | `configs/m3.yaml` → `inference.threshold` |
| Threshold selection rule | `max_balanced_accuracy` | `configs/m3.yaml` |
| Threshold selected on | `val` split | `configs/m3.yaml` |
| Embedding model | FaceNet InceptionResnetV1 (VGGFace2) | `configs/m3.yaml` |
| Embedding dim | 512 | `configs/m3.yaml` |
| Confidence formula | `sigmoid(10 × (score − threshold))` | `src/inference.py` |
| Dataset | LFW via TensorFlow Datasets | `configs/m3.yaml` |
| Split policy | 70% train / 15% val / 15% test (by identity) | `configs/m3.yaml` |
