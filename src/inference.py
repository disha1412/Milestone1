import time
from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.embeddings import preprocess_image, extract_embedding, get_model
from src.similarity import cosine_similarity_vectorized


def apply_threshold(score: float, threshold: float) -> int:
    return 1 if score >= threshold else 0


def compute_confidence(score: float, threshold: float) -> float:
    margin = score - threshold
    confidence = 1.0 / (1.0 + np.exp(-10.0 * margin))
    return float(np.clip(confidence, 0.0, 1.0))


def run_pair_inference(
    left_image,
    right_image,
    threshold: float,
    model_backend=None,
) -> Dict[str, Any]:
    if model_backend is None:
        model_backend = get_model()

    t_start = time.perf_counter()

    t0 = time.perf_counter()
    left_arr = preprocess_image(left_image)
    right_arr = preprocess_image(right_image)
    t_preprocess = time.perf_counter() - t0

    t0 = time.perf_counter()
    left_emb = extract_embedding(left_arr, model_backend)
    right_emb = extract_embedding(right_arr, model_backend)
    t_embedding = time.perf_counter() - t0

    t0 = time.perf_counter()
    a = left_emb.reshape(1, -1)
    b = right_emb.reshape(1, -1)
    score = float(cosine_similarity_vectorized(a, b)[0])
    t_scoring = time.perf_counter() - t0

    decision = apply_threshold(score, threshold)
    confidence = compute_confidence(score, threshold)

    total_latency = time.perf_counter() - t_start

    return {
        "score": score,
        "threshold": threshold,
        "decision": decision,
        "confidence": confidence,
        "latency_total_s": total_latency,
        "latency_preprocess_s": t_preprocess,
        "latency_embedding_s": t_embedding,
        "latency_scoring_s": t_scoring,
    }