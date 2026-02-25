"""
Ingest the LFW dataset using TensorFlow Datasets.
Produces a deterministic split by identity and writes a manifest file.
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ingest(config: dict) -> dict:
    import tensorflow_datasets as tfds

    seed = config["seed"]
    cache_dir = config["data"]["cache_dir"]
    split_ratios = config["split_policy"]
    manifest_path = config["manifest"]["output_path"]

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manual_dir = os.path.join(cache_dir, "downloads", "manual")
    os.makedirs(manual_dir, exist_ok=True)

    print(f"[ingest] Loading LFW from TFDS (cache: {cache_dir}) ...")
    builder = tfds.builder(config["data"]["tfds_name"], data_dir=cache_dir)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir=manual_dir)
    )
    ds = builder.as_dataset(split="train", shuffle_files=False)
    info = builder.info

    # Build identity -> list of image records
    # TFDS LFW exposes 'label' as bytes (person name), decode to string
    identity_map = defaultdict(list)
    for record in tfds.as_numpy(ds):
        label = record["label"].decode("utf-8") if isinstance(record["label"], bytes) else str(record["label"])
        identity_map[label].append(record)

    # Sort identity keys (strings = person names) for determinism
    sorted_identities = sorted(identity_map.keys())
    n_identities = len(sorted_identities)
    print(f"[ingest] Found {n_identities} identities.")

    # Deterministic shuffle of identities using fixed seed
    rng = random.Random(seed)
    shuffled = sorted_identities[:]
    rng.shuffle(shuffled)

    # Split identities
    n_train = int(n_identities * split_ratios["train"])
    n_val = int(n_identities * split_ratios["val"])

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train: n_train + n_val])
    test_ids = set(shuffled[n_train + n_val:])

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    # Count images per split
    counts = {}
    for split_name, id_set in splits.items():
        n_images = sum(len(identity_map[i]) for i in id_set)
        counts[split_name] = {
            "identities": len(id_set),
            "images": n_images,
        }
        print(f"  {split_name}: {len(id_set)} identities, {n_images} images")

    # Save split membership to disk so pair generation can reuse it
    splits_path = os.path.join(os.path.dirname(manifest_path), "splits.json")
    serializable_splits = {k: sorted(list(v)) for k, v in splits.items()}
    with open(splits_path, "w") as f:
        json.dump(serializable_splits, f, indent=2)
    print(f"[ingest] Split membership saved to {splits_path}")

    # Build manifest
    manifest = {
        "seed": seed,
        "split_policy": split_ratios["description"],
        "counts": counts,
        "data_source": {
            "tfds_name": config["data"]["tfds_name"],
            "cache_dir": os.path.abspath(cache_dir),
        },
        "splits_file": os.path.abspath(splits_path),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ingest] Manifest written to {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Ingest LFW dataset")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    ingest(config)
    print("[ingest] Done.")


if __name__ == "__main__":
    main()