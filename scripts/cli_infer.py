import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference import run_pair_inference
from src.embeddings import get_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_threshold(config: dict) -> float:
    t = config.get("inference", {}).get("threshold")
    if t is None:
        raise ValueError("No threshold set in config inference.threshold")
    return float(t)


def _is_record_key(path: str) -> bool:
    return path.startswith("lfw_record_")


def _build_lfw_image_map(config: dict) -> dict:
    import tensorflow_datasets as tfds
    cache_dir = config["data"]["cache_dir"]
    ds = tfds.load(config["data"]["tfds_name"], split="train", data_dir=cache_dir, shuffle_files=False)
    image_map = {}
    for i, record in enumerate(tfds.as_numpy(ds)):
        label = record["label"].decode("utf-8") if isinstance(record["label"], bytes) else str(record["label"])
        key = f"lfw_record_{label}_{i:06d}"
        image_map[key] = record["image"]
    return image_map


def _resolve_image(path: str, image_map: dict) -> np.ndarray:
    if _is_record_key(path):
        img = image_map.get(path)
        if img is None:
            raise KeyError(f"Record key not found in LFW map: {path}")
        return img
    return path


def print_result(pair_id: str, left: str, right: str, result: dict) -> None:
    decision_str = "SAME" if result["decision"] == 1 else "DIFFERENT"
    print(f"Pair:       {pair_id}")
    print(f"Left:       {left}")
    print(f"Right:      {right}")
    print(f"Score:      {result['score']:.6f}")
    print(f"Threshold:  {result['threshold']:.6f}")
    print(f"Decision:   {decision_str}")
    print(f"Confidence: {result['confidence']:.6f}")
    print(f"Latency:    {result['latency_total_s']*1000:.2f} ms")
    print()


def run_single(args, config: dict, threshold: float, model_backend) -> None:
    image_map = {}
    if _is_record_key(args.left) or _is_record_key(args.right):
        print("[cli] Detected LFW record keys — loading TFDS image map ...")
        image_map = _build_lfw_image_map(config)

    left_img = _resolve_image(args.left, image_map)
    right_img = _resolve_image(args.right, image_map)

    result = run_pair_inference(left_img, right_img, threshold=threshold, model_backend=model_backend)
    print_result("pair_0", args.left, args.right, result)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[cli] Result saved to {args.output}")


def run_batch(args, config: dict, threshold: float, model_backend) -> None:
    with open(args.pairs_file) as f:
        reader = csv.DictReader(f)
        pairs = list(reader)

    needs_lfw = any(
        _is_record_key(row["left_path"]) or _is_record_key(row["right_path"])
        for row in pairs
    )
    image_map = {}
    if needs_lfw:
        print("[cli] Detected LFW record keys — loading TFDS image map ...")
        image_map = _build_lfw_image_map(config)

    results = []
    for i, row in enumerate(pairs):
        pair_id = row.get("pair_id", f"pair_{i}")
        left_key = row["left_path"]
        right_key = row["right_path"]
        left_img = _resolve_image(left_key, image_map)
        right_img = _resolve_image(right_key, image_map)
        result = run_pair_inference(left_img, right_img, threshold=threshold, model_backend=model_backend)
        print_result(pair_id, left_key, right_key, result)
        results.append({"pair_id": pair_id, "left": left_key, "right": right_key, **result})

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[cli] {len(results)} results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Face verification CLI")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("pair", help="Infer on a single pair of images")
    p_single.add_argument("--left", required=True, help="Path to image file or LFW record key")
    p_single.add_argument("--right", required=True, help="Path to image file or LFW record key")
    p_single.add_argument("--output", default=None, help="Optional JSON output path")

    p_batch = sub.add_parser("batch", help="Infer on a CSV of pairs")
    p_batch.add_argument("--pairs-file", required=True, help="CSV with left_path,right_path columns")
    p_batch.add_argument("--output", default=None, help="Optional JSON output path")

    args = parser.parse_args()
    config = load_config(args.config)
    threshold = load_threshold(config)

    print(f"[cli] Loading model ...")
    model_backend = get_model()
    print(f"[cli] Model backend: {model_backend[0]}")
    print()

    if args.mode == "pair":
        run_single(args, config, threshold, model_backend)
    elif args.mode == "batch":
        run_batch(args, config, threshold, model_backend)


if __name__ == "__main__":
    main()