"""
Benchmark: Python loop vs NumPy vectorized similarity.
Reports timing and correctness checks for cosine and Euclidean.
"""

import argparse
import os
import time

import numpy as np
import yaml

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.similarity import (
    cosine_similarity_vectorized,
    euclidean_distance_vectorized,
    cosine_similarity_loop,
    euclidean_distance_loop,
)

TOLERANCE = 1e-6


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_benchmark(n: int, d: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    a = rng.random((n, d)).astype(np.float64)
    b = rng.random((n, d)).astype(np.float64)

    results = {}

    # --- Cosine ---
    t0 = time.perf_counter()
    cos_loop = cosine_similarity_loop(a, b)
    loop_cos_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    cos_vec = cosine_similarity_vectorized(a, b)
    vec_cos_time = time.perf_counter() - t0

    cos_max_diff = float(np.max(np.abs(cos_loop - cos_vec)))
    cos_correct = cos_max_diff < TOLERANCE

    results["cosine"] = {
        "loop_time_s": round(loop_cos_time, 6),
        "vec_time_s": round(vec_cos_time, 6),
        "speedup": round(loop_cos_time / vec_cos_time, 2) if vec_cos_time > 0 else None,
        "max_abs_diff": cos_max_diff,
        "correctness_pass": cos_correct,
    }

    # --- Euclidean ---
    t0 = time.perf_counter()
    euc_loop = euclidean_distance_loop(a, b)
    loop_euc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    euc_vec = euclidean_distance_vectorized(a, b)
    vec_euc_time = time.perf_counter() - t0

    euc_max_diff = float(np.max(np.abs(euc_loop - euc_vec)))
    euc_correct = euc_max_diff < TOLERANCE

    results["euclidean"] = {
        "loop_time_s": round(loop_euc_time, 6),
        "vec_time_s": round(vec_euc_time, 6),
        "speedup": round(loop_euc_time / vec_euc_time, 2) if vec_euc_time > 0 else None,
        "max_abs_diff": euc_max_diff,
        "correctness_pass": euc_correct,
    }

    return results


def format_report(results: dict, n: int, d: int) -> str:
    lines = [
        "=" * 55,
        f"  Similarity Benchmark  (N={n}, D={d})",
        "=" * 55,
    ]
    for metric, r in results.items():
        lines.append(f"\n[{metric.upper()}]")
        lines.append(f"  Loop time      : {r['loop_time_s']:.6f} s")
        lines.append(f"  Vectorized time: {r['vec_time_s']:.6f} s")
        lines.append(f"  Speedup        : {r['speedup']}x")
        lines.append(f"  Max abs diff   : {r['max_abs_diff']:.2e}")
        lines.append(f"  Correctness    : {'PASS' if r['correctness_pass'] else 'FAIL'}")
    lines.append("\n" + "=" * 55)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark similarity functions")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    n = config["benchmark"]["n_vectors"]
    d = config["benchmark"]["vector_dim"]
    seed = config["seed"]
    out_path = config["benchmark"]["output_path"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[bench] Running benchmark: N={n}, D={d}, seed={seed}")
    results = run_benchmark(n, d, seed)
    report = format_report(results, n, d)

    print(report)
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"[bench] Results saved to {out_path}")


if __name__ == "__main__":
    main()