"""
Hardware-aware profiling script for the face verification pipeline.
Measures preprocessing, embedding, and scoring latency separately.
Also runs batch-size sensitivity analysis.

Usage:
  python scripts/profile_latency.py --config configs/m3.yaml
  python scripts/profile_latency.py --config configs/m3.yaml --n-repeats 50 --output outputs/profiling/results.json
"""

import argparse
import json
import os
import platform
import sys
import time
from typing import List, Dict

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.embeddings import preprocess_image, extract_embedding, get_model
from src.similarity import cosine_similarity_vectorized


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False
    try:
        import tensorflow as tf
        info["tensorflow_version"] = tf.__version__
    except ImportError:
        info["tensorflow_version"] = "not installed"
    return info


def make_random_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)


def profile_single_pair(model_backend, n_repeats: int = 30) -> dict:
    """Profile a single pair inference with stage-level timing."""
    img_a = make_random_image(seed=0)
    img_b = make_random_image(seed=1)

    preprocess_times = []
    embedding_times = []
    scoring_times = []
    total_times = []

    # Warm-up
    for _ in range(3):
        arr = preprocess_image(img_a)
        _ = extract_embedding(arr, model_backend)

    for i in range(n_repeats):
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        arr_a = preprocess_image(img_a)
        arr_b = preprocess_image(img_b)
        preprocess_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        emb_a = extract_embedding(arr_a, model_backend)
        emb_b = extract_embedding(arr_b, model_backend)
        embedding_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        a = emb_a.reshape(1, -1)
        b = emb_b.reshape(1, -1)
        _ = float(cosine_similarity_vectorized(a, b)[0])
        scoring_times.append(time.perf_counter() - t0)

        total_times.append(time.perf_counter() - t_total_start)

    def stats(arr):
        a = np.array(arr) * 1000  # convert to ms
        return {
            "mean_ms": round(float(a.mean()), 4),
            "median_ms": round(float(np.median(a)), 4),
            "std_ms": round(float(a.std()), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4),
            "min_ms": round(float(a.min()), 4),
            "max_ms": round(float(a.max()), 4),
        }

    return {
        "n_repeats": n_repeats,
        "preprocessing": stats(preprocess_times),
        "embedding": stats(embedding_times),
        "scoring": stats(scoring_times),
        "total_end_to_end": stats(total_times),
    }


def profile_batch_sizes(model_backend, batch_sizes: List[int], n_repeats: int = 10) -> List[dict]:
    """Profile how latency and throughput change with batch size."""
    results = []
    for bs in batch_sizes:
        images = [make_random_image(seed=i) for i in range(bs)]

        batch_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            arrays = [preprocess_image(img) for img in images]
            embeddings = np.stack([extract_embedding(arr, model_backend) for arr in arrays])
            # Score all consecutive pairs
            if bs >= 2:
                a = embeddings[:-1]
                b = embeddings[1:]
                _ = cosine_similarity_vectorized(a, b)
            elapsed = time.perf_counter() - t0
            batch_times.append(elapsed)

        arr = np.array(batch_times) * 1000
        throughput = bs / np.mean(batch_times)
        results.append({
            "batch_size": bs,
            "total_mean_ms": round(float(arr.mean()), 3),
            "total_p95_ms": round(float(np.percentile(arr, 95)), 3),
            "per_image_mean_ms": round(float(arr.mean() / bs), 3),
            "throughput_images_per_s": round(float(throughput), 2),
        })
        print(f"  batch_size={bs:3d}  total={arr.mean():.1f}ms  per_image={arr.mean()/bs:.2f}ms  throughput={throughput:.1f} img/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Hardware-aware profiling for face verification pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--n-repeats", type=int, default=30, help="Number of timing repetitions for single-pair profiling")
    parser.add_argument("--batch-repeats", type=int, default=10, help="Number of timing repetitions for batch profiling")
    parser.add_argument("--output", default="outputs/profiling/results.json", help="Output JSON path")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("[profile] Collecting hardware info ...")
    hw = get_hardware_info()
    for k, v in hw.items():
        print(f"  {k}: {v}")

    print(f"\n[profile] Loading model backend ...")
    model_backend = get_model()
    backend_type = model_backend[0]
    print(f"  backend: {backend_type}")

    print(f"\n[profile] Single-pair stage profiling (n_repeats={args.n_repeats}) ...")
    single_pair_results = profile_single_pair(model_backend, n_repeats=args.n_repeats)

    sp = single_pair_results
    print(f"\n  Stage Latency Summary (mean ± std):")
    print(f"  Preprocessing : {sp['preprocessing']['mean_ms']:.3f} ms ± {sp['preprocessing']['std_ms']:.3f}")
    print(f"  Embedding     : {sp['embedding']['mean_ms']:.3f} ms ± {sp['embedding']['std_ms']:.3f}")
    print(f"  Scoring       : {sp['scoring']['mean_ms']:.4f} ms ± {sp['scoring']['std_ms']:.4f}")
    print(f"  End-to-end    : {sp['total_end_to_end']['mean_ms']:.3f} ms ± {sp['total_end_to_end']['std_ms']:.3f}")

    total = sp['total_end_to_end']['mean_ms']
    if total > 0:
        print(f"\n  Stage breakdown (% of total):")
        print(f"  Preprocessing : {100*sp['preprocessing']['mean_ms']/total:.1f}%")
        print(f"  Embedding     : {100*sp['embedding']['mean_ms']/total:.1f}%")
        print(f"  Scoring       : {100*sp['scoring']['mean_ms']/total:.2f}%")

    batch_sizes = [1, 2, 4, 8, 16, 32]
    print(f"\n[profile] Batch-size sensitivity (repeats={args.batch_repeats}) ...")
    batch_results = profile_batch_sizes(model_backend, batch_sizes, n_repeats=args.batch_repeats)

    report = {
        "hardware": hw,
        "model_backend": backend_type,
        "single_pair_profiling": single_pair_results,
        "batch_size_sensitivity": batch_results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[profile] Results saved to {args.output}")


if __name__ == "__main__":
    main()
