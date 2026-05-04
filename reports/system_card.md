# System Card — Face Verification Pipeline
**Version:** v1.0-final  
**Milestone:** 4 (Final Release)  
**Course:** MSML/MSAI 605  
**Date:** 2025

---

## 1. System Overview

This system is a face verification pipeline that accepts two face images and
returns a binary decision — **SAME** (same person) or **DIFFERENT** (different
people) — along with a calibrated confidence score and per-stage latency
measurements.

### Pipeline Summary

```
Image A, Image B
     │
     ▼
Preprocessing
  Resize to 160×160, per-image mean/std normalization
     │
     ▼
Embedding Extraction
  FaceNet InceptionResnetV1 pretrained on VGGFace2 (512-dim)
     │
     ▼
Cosine Similarity Score  ∈ [-1, 1]
     │
     ▼
Threshold Decision
  score >= 0.35 → SAME (label=1)
  score <  0.35 → DIFFERENT (label=0)
     │
     ▼
Calibrated Confidence
  sigmoid(10 × (score − 0.35))  ∈ [0, 1]
  0.5 = exactly on the decision boundary
```

### Inputs

- Two RGB face images in any format supported by Pillow (JPEG, PNG, etc.)
- Images are internally resized to 160×160; no prior alignment or cropping is required, though better-cropped face inputs will improve reliability.

### Outputs

| Field | Type | Description |
|---|---|---|
| `score` | float ∈ [-1, 1] | Cosine similarity between the two face embeddings |
| `decision` | int {0, 1} | 1 = SAME, 0 = DIFFERENT |
| `confidence` | float ∈ [0, 1] | Calibrated confidence; 0.5 = boundary |
| `threshold` | float | Operating threshold used (0.35) |
| `latency_total_s` | float | End-to-end wall time in seconds |
| `latency_preprocess_s` | float | Preprocessing stage time |
| `latency_embedding_s` | float | Embedding extraction stage time |
| `latency_scoring_s` | float | Cosine similarity computation time |

---

## 2. Intended Use

### Supported use cases

- **Academic research and coursework:** Evaluating face verification algorithms
  in a controlled, research setting with proper oversight.
- **Offline batch verification:** Processing pairs of face images in a pipeline
  where human review is part of the downstream workflow.
- **System benchmarking:** Comparing embedding-based versus pixel-based
  verification baselines under reproducible conditions.
- **Educational demonstrations:** Understanding the components of a face
  verification pipeline including threshold calibration, confidence scoring,
  and latency profiling.

### Out-of-scope uses

The following uses are explicitly **not supported** and should not be attempted
with this system:

- **Real-time surveillance or tracking** of individuals in public or private spaces.
- **Access control, authentication, or security systems** without extensive
  re-evaluation, re-calibration, and independent safety review.
- **Any application affecting legal rights, employment, housing, or law enforcement.**
- **High-stakes identity verification** in production environments (e.g., border
  control, financial verification).
- **Use on minors or in contexts involving children** without appropriate ethical
  review and consent processes.
- **Deployment on populations significantly different from the LFW training
  distribution** without re-evaluation on a representative held-out dataset.

---

## 3. Data Summary

### Dataset

- **Source:** Labeled Faces in the Wild (LFW), loaded via TensorFlow Datasets.
- **Split policy:** Identity-level split — 70% train / 15% val / 15% test.
  No identity appears in more than one split, preventing leakage.
- **Pair generation:** 500 positive (same-identity) and 500 negative
  (different-identity) pairs per split, generated deterministically using a
  fixed seed.

### Known data limitations

- **Celebrity/public figure bias:** LFW consists largely of images of public
  figures collected from news media. The demographic distribution skews toward
  certain age groups, ethnicities, and professions, and does not represent the
  general population.
- **Image quality variation:** Images vary in lighting, pose, resolution, and
  occlusion. The pipeline does not perform face alignment, which may affect
  reliability on low-quality or non-frontal inputs.
- **Class imbalance by identity:** Some identities have many more images than
  others. The Milestone 2 data-centric analysis introduced an optional cap
  (`max_images_per_identity`) to reduce dominance by overrepresented subjects.
- **No demographic metadata:** LFW does not provide ground-truth demographic
  labels. Fairness analysis is therefore limited to qualitative risk
  identification rather than quantitative subgroup metrics.

---

## 4. Operating Threshold and Key Metrics

### Threshold

| Parameter | Value |
|---|---|
| Operating threshold | **0.35** |
| Selection rule | Max balanced accuracy |
| Selected on | Validation split |
| Config location | `configs/m3.yaml` → `inference.threshold` |

The threshold of **0.35** was selected by sweeping 100 evenly spaced candidate
thresholds over the validation split and choosing the one that maximized
balanced accuracy, defined as the mean of true positive rate (TPR) and true
negative rate (1 - FPR).

### Key metrics at threshold = 0.35 (validation split, embedding-based system)

*Note: Metrics below reflect the embedding-based FaceNet system (Milestone 3
and 4 final version). The pixel-cosine baseline from Milestone 2 used a
different threshold and produces lower accuracy.*

| Metric | Value |
|---|---|
| Accuracy | Reported in `outputs/eval/val/metrics.json` |
| Balanced accuracy | Reported in `outputs/eval/val/metrics.json` |
| Precision | Reported in `outputs/eval/val/metrics.json` |
| Recall (TPR) | Reported in `outputs/eval/val/metrics.json` |
| F1 | Reported in `outputs/eval/val/metrics.json` |
| FPR | Reported in `outputs/eval/val/metrics.json` |

> To reproduce the exact numbers, follow the reproducibility checklist in
> `reports/reproducibility_checklist.md`.

### Confidence interpretation

| Confidence value | Meaning |
|---|---|
| 1.0 | Very high same-person confidence (score >> threshold) |
| > 0.5 | Same-person lean |
| 0.5 | Exactly on the decision boundary |
| < 0.5 | Different-person lean |
| 0.0 | Very high different-person confidence (score << threshold) |

---

## 5. Failure Modes and Limitations

### Known failure modes

**1. Low-quality or non-frontal images**  
The pipeline resizes to 160×160 without face detection or alignment. Heavily
occluded faces, extreme profile poses, or very low resolution inputs will
degrade embedding quality and produce unreliable scores.

**2. Near-boundary scores**  
Pairs with scores close to the threshold (within ±0.05) produce confidence
values near 0.5. These decisions are inherently uncertain and should be treated
as low-confidence rather than definitive.

**3. Visually similar different-identity pairs**  
The system can produce false positives when two different people are visually
similar (e.g., twins, family members with similar appearance, or individuals
photographed in very similar conditions).

**4. Non-face inputs**  
The pipeline does not include a face detector. Non-face images (e.g., objects,
text, backgrounds) will be embedded and compared, producing meaningless scores
without any error or warning.

**5. Model fallback degradation**  
If `facenet-pytorch` is not installed, the pipeline falls back to a TF Hub
MobileNetV2 feature extractor, and then to a normalized pixel vector. The
threshold of 0.35 was calibrated for the FaceNet backend. Using a fallback
backend with the same threshold is likely to produce degraded performance.

**6. Distribution shift**  
The system was calibrated on LFW. Performance may degrade on images from
substantially different distributions (different cameras, different
demographics, medical or forensic contexts).

---

## 6. Fairness-Related Risks and Misuse Concerns

### Important caveat

LFW does not include verified demographic labels. The analysis below is
qualitative. No subgroup performance numbers are claimed because the data does
not reliably support them.

### Identified risk categories

**Demographic representation imbalance**  
LFW is known to underrepresent certain demographic groups, particularly women
and individuals from some ethnic backgrounds. A model trained or calibrated
primarily on one demographic subgroup may generalize less reliably to others.
This risk exists even though this system uses a pretrained model (VGGFace2)
rather than training from scratch on LFW.

**Image quality disparities**  
If certain populations are more likely to be photographed under poor lighting,
at non-frontal angles, or with lower resolution, this system may perform less
reliably for those groups — not because of intentional bias, but because the
model was not calibrated for those conditions.

**Surveillance and tracking misuse**  
Face verification systems can be misused for surveillance, stalking, or
unauthorized identification. This system is not designed or validated for any
such use. The threshold and confidence calibration are tuned for a
research/academic setting, not for real-world identity decisions.

**False positive harms**  
In any high-stakes deployment (which is explicitly out of scope), false
positives — incorrectly matching two different people — could lead to serious
harm. At the selected threshold of 0.35 on LFW, false positives exist and are
documented in the error analysis artifacts (`outputs/eval/val/error_analysis.json`).

**False negative harms**  
False negatives — failing to match the same person — could also cause harm in
access control or verification settings. Both error types should be considered
before any deployment.

### Recommended mitigations (for any future deployment consideration)

- Evaluate on a demographically representative held-out dataset before any deployment.
- Use human review for near-boundary decisions (confidence near 0.5).
- Clearly document intended use and do not deploy for surveillance or legal purposes.
- Re-calibrate the threshold if the deployment distribution differs from LFW.

---

## 7. Operational Constraints

| Constraint | Details |
|---|---|
| Input format | Any RGB image readable by Pillow (JPEG, PNG, etc.) |
| Input size | Internally resized to 160×160; no minimum size enforced |
| Embedding model | FaceNet InceptionResnetV1 (facenet-pytorch) |
| Fallback behavior | TF Hub MobileNetV2 → normalized pixel vector if model unavailable |
| Threshold validity | Threshold 0.35 is only calibrated for the FaceNet backend |
| CPU baseline latency | See `reports/profiling_report.pdf` for measured values |
| GPU support | Optional; see profiling report for comparison if available |
| Concurrency | Tested with `ProcessPoolExecutor` (4 workers, 50 requests); see load test |
| Container | Docker image `face-verifier:v1.0-final` (see Dockerfile) |
| Python version | 3.10+ required |
| Key dependencies | See `requirements.txt` |

### Latency budget guidance

- Single pair, CPU, FaceNet backend: embedding stage typically dominates
  (see profiling report for exact measurements).
- For latency-sensitive applications, batch processing amortizes the
  per-image cost; see batch-size sensitivity in the profiling report.

---

## 8. Reproducibility Pointer

| Item | Location |
|---|---|
| Full How-to-run instructions | `README.md` |
| Exact reproduction commands | `reports/reproducibility_checklist.md` |
| Final inference config | `configs/m3.yaml` |
| Profiling report | `reports/profiling_report.pdf` |
| Final Git tag | `v1.0-final` |
| Docker entrypoint | `scripts/cli_infer.py` via `Dockerfile` |
| Test suite | `pytest tests/ -v` |

A grader can reproduce the core outputs by following the reproducibility
checklist exactly. No LFW data download is required to verify CLI behavior or
run the test suite (both work with the synthetic test images generated by
`make_test_imgs.py`).
