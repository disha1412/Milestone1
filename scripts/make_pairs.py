"""
Generate deterministic verification pairs (positive and negative) per split.
Reads the splits produced by ingest_lfw.py and writes CSV pair files.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

import yaml
import tensorflow_datasets as tfds


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_identity_map(config: dict) -> dict:
    """Load dataset and return {identity_label: [image_paths]}."""
    cache_dir = config["data"]["cache_dir"]
    ds = tfds.load(
        config["data"]["tfds_name"],
        split="train",
        data_dir=cache_dir,
        shuffle_files=False,
    )
    identity_map = defaultdict(list)
    for i, record in enumerate(tfds.as_numpy(ds)):
        label = record["label"].decode("utf-8") if isinstance(record["label"], bytes) else str(record["label"])
        # Use a stable string key as "path" (index-based for TFDS in-memory data)
        identity_map[label].append(f"lfw_record_{label}_{i:06d}")
    # Sort image lists within each identity for determinism
    for k in identity_map:
        identity_map[k].sort()
    return identity_map


def generate_pairs(identity_map: dict, id_set: list, n_pos: int, n_neg: int, seed: int, split_name: str) -> list:
    rng = random.Random(seed)
    rows = []

    # Sort the id_set for determinism
    id_set = sorted(id_set)

    # Positive pairs: same identity, need at least 2 images
    eligible = [i for i in id_set if len(identity_map[i]) >= 2]
    pos_count = 0
    # Cycle through eligible identities deterministically
    rng.shuffle(eligible)
    idx = 0
    while pos_count < n_pos and eligible:
        ident = eligible[idx % len(eligible)]
        imgs = identity_map[ident]
        a, b = rng.sample(imgs, 2)
        rows.append({"left_path": a, "right_path": b, "label": 1, "split": split_name})
        pos_count += 1
        idx += 1

    # Negative pairs: different identities
    neg_count = 0
    while neg_count < n_neg:
        id_a, id_b = rng.sample(id_set, 2)
        img_a = rng.choice(identity_map[id_a])
        img_b = rng.choice(identity_map[id_b])
        rows.append({"left_path": img_a, "right_path": img_b, "label": 0, "split": split_name})
        neg_count += 1

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate verification pairs")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"]
    n_pos = config["pairs"]["num_positive_per_split"]
    n_neg = config["pairs"]["num_negative_per_split"]
    output_dir = config["pairs"]["output_dir"]
    manifest_path = config["manifest"]["output_path"]

    os.makedirs(output_dir, exist_ok=True)

    # Load split membership from manifest directory
    splits_path = os.path.join(os.path.dirname(manifest_path), "splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)  # {split_name: [identity_ids]}

    print("[pairs] Building identity map from TFDS ...")
    identity_map = build_identity_map(config)

    fieldnames = ["left_path", "right_path", "label", "split"]

    for split_name, id_list in splits.items():
        # Use a split-specific seed offset for independence, but derived from main seed
        split_seed = seed + hash(split_name) % 10000
        rows = generate_pairs(identity_map, id_list, n_pos, n_neg, split_seed, split_name)

        out_path = os.path.join(output_dir, f"{split_name}_pairs.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[pairs] {split_name}: {len(rows)} pairs -> {out_path}")

    print("[pairs] Done.")


if __name__ == "__main__":
    main()