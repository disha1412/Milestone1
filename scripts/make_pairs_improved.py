"""
Data-centric improvement: regenerate pairs with a cap on images per identity.

Problem with baseline pairs:
  A small number of identities (e.g. George W. Bush with 530+ images) dominate
  the pair pool. This means most positive pairs come from a single person's images,
  making the positive set unrepresentative and biasing evaluation.

Change applied:
  Cap each identity to at most `improved_max_images_per_identity` images (default=5)
  before sampling pairs. This ensures no single identity contributes
  disproportionately, producing a more diverse and uniformly difficult positive set.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import tensorflow_datasets as tfds


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_identity_map_capped(config: dict, max_images: int) -> dict:
    """
    Load LFW from TFDS and build {identity: [record_ids]},
    capping each identity to at most max_images entries.
    """
    cache_dir = config["data"]["cache_dir"]
    ds = tfds.load(
        config["data"]["tfds_name"],
        split="train",
        data_dir=cache_dir,
        shuffle_files=False,
    )

    identity_map = defaultdict(list)
    for i, record in enumerate(tfds.as_numpy(ds)):
        label = (
            record["label"].decode("utf-8")
            if isinstance(record["label"], bytes)
            else str(record["label"])
        )
        identity_map[label].append(f"lfw_record_{label}_{i:06d}")

    # Sort each identity's list for determinism, then cap
    capped = {}
    n_capped = 0
    for ident, imgs in identity_map.items():
        imgs_sorted = sorted(imgs)
        if len(imgs_sorted) > max_images:
            capped[ident] = imgs_sorted[:max_images]
            n_capped += 1
        else:
            capped[ident] = imgs_sorted

    print(f"[improved pairs] Capped {n_capped} identities to max {max_images} images.")
    return capped


def generate_pairs(identity_map: dict, id_set: list, n_pos: int, n_neg: int,
                   seed: int, split_name: str) -> list:
    rng = random.Random(seed)
    rows = []
    id_set = sorted(id_set)

    # Positive pairs
    eligible = [i for i in id_set if len(identity_map[i]) >= 2]
    rng.shuffle(eligible)
    pos_count = 0
    idx = 0
    while pos_count < n_pos and eligible:
        ident = eligible[idx % len(eligible)]
        imgs = identity_map[ident]
        a, b = rng.sample(imgs, 2)
        rows.append({"left_path": a, "right_path": b, "label": 1, "split": split_name})
        pos_count += 1
        idx += 1

    # Negative pairs
    neg_count = 0
    while neg_count < n_neg:
        id_a, id_b = rng.sample(id_set, 2)
        img_a = rng.choice(identity_map[id_a])
        img_b = rng.choice(identity_map[id_b])
        rows.append({"left_path": img_a, "right_path": img_b, "label": 0, "split": split_name})
        neg_count += 1

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate improved pairs with identity cap")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"]
    max_images = config["pairs"]["improved_max_images_per_identity"]
    output_dir = config["pairs"]["improved_dir"]
    manifest_path = config.get("manifest", {}).get("output_path", "outputs/manifest.json")
    splits_path = os.path.join(os.path.dirname(manifest_path), "splits.json")

    os.makedirs(output_dir, exist_ok=True)

    with open(splits_path) as f:
        splits = json.load(f)

    # Use M1 pair counts from m1.yaml for consistency
    m1_config_path = os.path.join(os.path.dirname(args.config), "m1.yaml")
    with open(m1_config_path) as f:
        m1_config = yaml.safe_load(f)
    n_pos = m1_config["pairs"]["num_positive_per_split"]
    n_neg = m1_config["pairs"]["num_negative_per_split"]

    print(f"[improved pairs] Building capped identity map (max {max_images} images/identity) ...")
    identity_map = build_identity_map_capped(config, max_images)

    fieldnames = ["left_path", "right_path", "label", "split"]
    for split_name, id_list in splits.items():
        split_seed = seed + hash(split_name) % 10000
        rows = generate_pairs(identity_map, id_list, n_pos, n_neg, split_seed, split_name)
        out_path = os.path.join(output_dir, f"{split_name}_pairs.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[improved pairs] {split_name}: {len(rows)} pairs -> {out_path}")

    print("[improved pairs] Done.")


if __name__ == "__main__":
    main()
