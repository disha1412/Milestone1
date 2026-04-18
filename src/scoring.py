import time
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from typing import Dict, Tuple

from src.similarity import cosine_similarity_vectorized
from src.embeddings import preprocess_image, extract_embedding, get_model


def _load_lfw_images(config: dict) -> Dict[str, np.ndarray]:
    cache_dir = config["data"]["cache_dir"]
    ds = tfds.load(config["data"]["tfds_name"], split="train", data_dir=cache_dir, shuffle_files=False)
    image_map = {}
    for i, record in enumerate(tfds.as_numpy(ds)):
        label = record["label"].decode("utf-8") if isinstance(record["label"], bytes) else str(record["label"])
        key = f"lfw_record_{label}_{i:06d}"
        image_map[key] = record["image"]
    return image_map


def score_pairs(pairs_df: pd.DataFrame, config: dict) -> np.ndarray:
    image_map = _load_lfw_images(config)
    model_backend = get_model()

    left_embs, right_embs = [], []
    for _, row in pairs_df.iterrows():
        limg = image_map.get(row["left_path"])
        rimg = image_map.get(row["right_path"])
        lv = extract_embedding(limg, model_backend) if limg is not None else np.zeros(1)
        rv = extract_embedding(rimg, model_backend) if rimg is not None else np.zeros(1)
        left_embs.append(lv)
        right_embs.append(rv)

    d = max(v.shape[0] for v in left_embs + right_embs)

    def pad(v):
        if v.shape[0] < d:
            return np.concatenate([v, np.zeros(d - v.shape[0])])
        return v

    a = np.stack([pad(v) for v in left_embs])
    b = np.stack([pad(v) for v in right_embs])
    return cosine_similarity_vectorized(a, b)


def score_pair_timed(left_image, right_image, model_backend=None) -> Tuple[float, float]:
    t0 = time.perf_counter()
    if model_backend is None:
        model_backend = get_model()
    left_arr = preprocess_image(left_image)
    right_arr = preprocess_image(right_image)
    left_emb = extract_embedding(left_arr, model_backend)
    right_emb = extract_embedding(right_arr, model_backend)
    a = left_emb.reshape(1, -1)
    b = right_emb.reshape(1, -1)
    score = float(cosine_similarity_vectorized(a, b)[0])
    latency = time.perf_counter() - t0
    return score, latency