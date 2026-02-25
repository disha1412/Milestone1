# 605-Milestone1-Vansh-and-Disha

# MSML 605 — Milestone 1

This milestone builds the reproducible, deterministic plumbing that later milestones
will reuse: LFW dataset ingestion, deterministic identity-based splits, verification
pair generation, and vectorized similarity scoring with benchmarks.

---

## Repo Layout

```
repo_root/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── m1.yaml          # seed, paths, split and pair policies
├── src/
│   └── similarity.py    # vectorized cosine & Euclidean (importable module)
├── scripts/
│   ├── ingest_lfw.py    # download/load LFW, split by identity, write manifest
│   ├── make_pairs.py    # generate deterministic pair CSV files per split
│   └── bench_similarity.py  # loop vs vectorized benchmark + correctness check
├── tests/               # (optional unit tests)
├── data/                # dataset cache — NOT committed (see .gitignore)
└── outputs/             # generated artifacts — NOT committed (see .gitignore)
```

---

## How to Run

### 1. Create and activate environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### 2. Ingest LFW and write manifest
```bash
python scripts/ingest_lfw.py --config configs/m1.yaml
```
Produces: `outputs/manifest.json`, `outputs/splits.json`

---

### 3. Generate verification pairs
```bash
python scripts/make_pairs.py --config configs/m1.yaml
```
Produces: `outputs/pairs/train_pairs.csv`, `outputs/pairs/val_pairs.csv`, `outputs/pairs/test_pairs.csv`

---

### 4. Run similarity benchmark
```bash
python scripts/bench_similarity.py --config configs/m1.yaml
```
Produces: `outputs/bench/results.txt` (also printed to console)

---

## Outputs Summary

| File | Description |
|---|---|
| `outputs/manifest.json` | Seed, split policy, counts per split, data source |
| `outputs/splits.json` | Identity IDs assigned to each split |
| `outputs/pairs/{split}_pairs.csv` | Verification pairs: left_path, right_path, label, split |
| `outputs/bench/results.txt` | Loop vs vectorized timing + correctness check |

---

## Determinism Notes

- **Seed:** `42` (set in `configs/m1.yaml`)
- **Ingestion:** Identities are sorted before shuffling; shuffle uses `random.Random(seed)`.
- **Pairs:** Candidate lists are sorted before sampling; each split uses a derived seed.
- **Benchmark:** Vectors are generated with `np.random.default_rng(seed)`.
- Running any script twice with the same config produces byte-identical outputs.

---

## Git Tag

This milestone is tagged `v0.1`.  
To reproduce from a clean clone:
```bash
git clone <repo_url>
git checkout v0.1
# then follow the "How to Run" steps above
```