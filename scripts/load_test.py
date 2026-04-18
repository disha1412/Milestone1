import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _worker_infer(task: dict) -> dict:
    from src.inference import run_pair_inference
    from src.embeddings import get_model
    model_backend = get_model()
    try:
        result = run_pair_inference(
            task["left_path"],
            task["right_path"],
            threshold=task["threshold"],
            model_backend=model_backend,
        )
        return {"success": True, "latency_s": result["latency_total_s"], "pair_id": task["pair_id"]}
    except Exception as e:
        return {"success": False, "latency_s": None, "pair_id": task["pair_id"], "error": str(e)}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_tasks(pairs_path: str, threshold: float, n_requests: int) -> List[dict]:
    with open(pairs_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    tasks = []
    for i in range(n_requests):
        row = rows[i % len(rows)]
        tasks.append({
            "pair_id": f"req_{i:05d}",
            "left_path": row["left_path"],
            "right_path": row["right_path"],
            "threshold": threshold,
        })
    return tasks


def run_load_test(
    tasks: List[dict],
    n_workers: int,
    output_path: str,
) -> dict:
    print(f"[load_test] Running {len(tasks)} requests with {n_workers} workers ...")
    latencies = []
    failures = 0

    t_wall_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker_infer, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res["success"]:
                latencies.append(res["latency_s"])
            else:
                failures += 1
    t_wall_total = time.perf_counter() - t_wall_start

    n_success = len(latencies)
    n_total = len(tasks)
    throughput = n_success / t_wall_total if t_wall_total > 0 else 0.0

    latencies_arr = np.array(latencies) if latencies else np.array([0.0])
    summary = {
        "total_requests": n_total,
        "successful": n_success,
        "failures": failures,
        "n_workers": n_workers,
        "wall_time_s": round(t_wall_total, 4),
        "throughput_rps": round(throughput, 4),
        "latency_mean_ms": round(float(latencies_arr.mean()) * 1000, 3),
        "latency_median_ms": round(float(np.median(latencies_arr)) * 1000, 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)) * 1000, 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)) * 1000, 3),
        "latency_max_ms": round(float(latencies_arr.max()) * 1000, 3),
        "latency_min_ms": round(float(latencies_arr.min()) * 1000, 3),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[load_test] Done.")
    print(f"  Total requests : {n_total}")
    print(f"  Successful     : {n_success}")
    print(f"  Failures       : {failures}")
    print(f"  Wall time      : {t_wall_total:.2f}s")
    print(f"  Throughput     : {throughput:.2f} req/s")
    print(f"  Latency mean   : {summary['latency_mean_ms']:.1f} ms")
    print(f"  Latency p95    : {summary['latency_p95_ms']:.1f} ms")
    print(f"  Latency p99    : {summary['latency_p99_ms']:.1f} ms")
    print(f"  Results saved  : {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Concurrency / load test for face verification inference")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--pairs-file", required=True, help="CSV with left_path,right_path columns")
    parser.add_argument("--n-requests", type=int, default=None, help="Total number of requests (default: from config)")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of parallel workers (default: from config)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: from config)")
    args = parser.parse_args()

    config = load_config(args.config)
    lt_cfg = config.get("load_test", {})

    threshold = config.get("inference", {}).get("threshold")
    if threshold is None:
        raise ValueError("Config must have inference.threshold set")
    threshold = float(threshold)

    n_requests = args.n_requests or lt_cfg.get("n_requests", 50)
    n_workers = args.n_workers or lt_cfg.get("n_workers", 4)
    output_path = args.output or lt_cfg.get("output_path", "outputs/load_test/results.json")

    tasks = build_tasks(args.pairs_file, threshold, n_requests)
    run_load_test(tasks, n_workers, output_path)


if __name__ == "__main__":
    main()