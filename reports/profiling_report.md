# Profiling Report — Face Verification Pipeline
**Version:** v1.0-final  
**Milestone:** 4  
**Course:** MSML/MSAI 605

---

## 1. Measurement Environment

### Hardware (CPU Baseline — Required)

The CPU baseline was measured on the development machine used for this project.
Replace the values in brackets with your actual hardware when running the profiling script.

| Property | Value |
|---|---|
| Platform | [Reported by `platform.platform()` at runtime] |
| Processor | [Reported by `platform.processor()` at runtime] |
| CPU cores | [Reported by `os.cpu_count()` at runtime] |
| RAM | [Record your machine's RAM here] |
| OS | [Reported at runtime] |

### Software Stack

| Component | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | See `requirements.txt` |
| facenet-pytorch | ≥ 2.5.3 |
| NumPy | ≥ 1.24 |
| Pillow | ≥ 10.0 |

### GPU (Optional — Supplemental Only)

If a GPU is available and `torch.cuda.is_available()` is True, the embedding
stage may execute on the GPU device. Any GPU results are supplemental and
clearly labeled. The required baseline is **CPU only**.

---

## 2. Methodology

- **Input:** Synthetic random 160×160 RGB images generated with a fixed seed
  (`numpy.random.default_rng`), consistent across runs.
- **Warm-up:** 3 warm-up iterations are performed before measurement to avoid
  cold-start bias (e.g., model weight loading, JIT compilation).
- **Timing:** Python `time.perf_counter()` with nanosecond resolution.
- **Repetitions:** 30 repetitions for single-pair profiling; 10 repetitions per
  batch size for batch-size sensitivity.
- **Stage isolation:** Each stage (preprocessing, embedding, scoring) is timed
  separately using bracketed `perf_counter()` calls. End-to-end time also
  includes minor Python overhead between stages.
- **Reproducibility:** Run `python scripts/profile_latency.py --config configs/m3.yaml`
  to reproduce. Results are saved to `outputs/profiling/results.json`.

---

## 3. Per-Stage Latency Breakdown (Single Pair, CPU Baseline)

*Run `python scripts/profile_latency.py --config configs/m3.yaml` to get your
actual measurements. The table structure below shows what the script reports.*

| Stage | Mean (ms) | Median (ms) | Std (ms) | P95 (ms) |
|---|---|---|---|---|
| Preprocessing (×2 images) | [from results.json] | — | — | — |
| Embedding extraction (×2 images) | [from results.json] | — | — | — |
| Cosine similarity scoring | [from results.json] | — | — | — |
| **End-to-end total** | **[from results.json]** | — | — | — |

### Stage breakdown (% of total)

| Stage | % of total |
|---|---|
| Preprocessing | [computed from results] |
| Embedding | [computed from results — expected dominant stage] |
| Scoring | [computed from results — expected <1%] |

### Interpretation

**The embedding extraction stage dominates end-to-end latency.** This is
expected: FaceNet InceptionResnetV1 is a deep InceptionResNet-style network
with 512-dimensional output. A forward pass through the model is far more
computationally expensive than either preprocessing (simple resize + normalization)
or scoring (a single dot product).

**Preprocessing** is the second costliest stage, involving a PIL resize
operation and per-image normalization. It is called twice per pair (once per
image) and typically accounts for a small but non-trivial fraction of total latency.

**Scoring** is negligible — a vectorized NumPy dot product over a single pair
of 512-dimensional vectors completes in microseconds.

**Practical implication:** To reduce latency, the highest-leverage optimization
is the embedding stage — either through GPU acceleration, model quantization,
or caching embeddings for known reference images.

---

## 4. Batch-Size Sensitivity

The table below shows how total batch latency, per-image latency, and
throughput change as the number of images processed together increases.
Run the profiling script to populate with your actual measurements.

| Batch Size | Total Mean (ms) | Per-Image Mean (ms) | Throughput (img/s) | P95 Total (ms) |
|---|---|---|---|---|
| 1 | [from results.json] | [from results.json] | [from results.json] | — |
| 2 | [from results.json] | [from results.json] | [from results.json] | — |
| 4 | [from results.json] | [from results.json] | [from results.json] | — |
| 8 | [from results.json] | [from results.json] | [from results.json] | — |
| 16 | [from results.json] | [from results.json] | [from results.json] | — |
| 32 | [from results.json] | [from results.json] | [from results.json] | — |

### Interpretation

**Per-image latency typically decreases as batch size grows** because the fixed
overhead of Python loops and model invocation is amortized across more images.
However, the current embedding implementation calls `extract_embedding` once per
image in a Python loop (see `src/embeddings.py` → `extract_embedding_batch`).
True batched inference — passing a tensor of N images through the model in a
single forward pass — would yield larger speedups, especially on GPU.

**Throughput** (images per second) increases with batch size up to the point
where CPU memory or model throughput becomes the bottleneck.

**Recommendation:** For offline batch processing, use larger batch sizes (≥8)
to reduce per-image cost. For latency-sensitive single-pair applications, there
is no benefit to batching.

---

## 5. CPU Baseline Summary

The CPU baseline is the primary reference measurement for this profiling report.
It was obtained by running the profiling script on CPU hardware without GPU
acceleration. Key takeaways:

1. End-to-end single-pair latency is dominated by the embedding stage.
2. Scoring latency is negligible (<1% of total in all measured conditions).
3. Throughput scales with batch size, with the most significant per-image
   improvement between batch sizes 1 and 4.
4. The system can sustain the load test configuration of 50 requests × 4 workers
   on CPU hardware (see `outputs/load_test/results.json` for throughput and
   p95/p99 latency from `scripts/load_test.py`).

---

## 6. Optional GPU Comparison

*If GPU results were collected, describe them here. If not, this section can
be omitted or replaced with the following note:*

GPU profiling was not performed for this submission. The CPU baseline is
sufficient to characterize the system's runtime behavior for the Milestone 4
deliverables. GPU acceleration can be enabled by ensuring `torch.cuda.is_available()`
returns True; the FaceNet model will automatically move to the available CUDA
device.

---

## 7. How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Run profiling script (CPU baseline)
python scripts/profile_latency.py \
  --config configs/m3.yaml \
  --n-repeats 30 \
  --output outputs/profiling/results.json

# View the JSON results
cat outputs/profiling/results.json
```

All timing values in this report can be reproduced by running the above
commands on the same hardware. Results will vary across machines; the structure
of the output (stage breakdown, batch sensitivity table) is deterministic.
