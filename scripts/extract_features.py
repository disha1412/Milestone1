"""
Extract image features from the LFW TFDS cache.
Produces outputs/features.npz — a compressed map of record_id -> feature vector.
Run once; subsequent evaluation runs reuse this file.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.features import build_feature_map, save_features


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Extract LFW image features")
    parser.add_argument("--config", required=True, help="Path to m2.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = config["features"]["output_path"]

    if os.path.exists(output_path):
        print(f"[extract] Features already exist at {output_path}. Delete to re-extract.")
        return

    features = build_feature_map(config)
    save_features(features, output_path)
    print("[extract] Done.")


if __name__ == "__main__":
    main()
