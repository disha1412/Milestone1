import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Optional


TARGET_SIZE = (160, 160)
EMBEDDING_DIM = 512


def preprocess_image(image_input, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    if isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype(np.uint8))
    elif isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    mean, std = arr.mean(), arr.std()
    if std < 1e-6:
        std = 1e-6
    return (arr - mean) / std


def _load_facenet_model(model_path: Optional[str] = None):
    try:
        from facenet_pytorch import InceptionResnetV1
        import torch
        model = InceptionResnetV1(pretrained="vggface2").eval()
        return ("pytorch", model)
    except ImportError:
        pass
    try:
        import tensorflow_hub as hub
        url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
        model = hub.load(url)
        return ("tfhub", model)
    except Exception:
        pass
    return ("mock", None)


_MODEL_CACHE = None


def get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _load_facenet_model()
    return _MODEL_CACHE


def extract_embedding(image_input, model_backend=None) -> np.ndarray:
    if model_backend is None:
        model_backend = get_model()
    backend_type, model = model_backend

    arr = preprocess_image(image_input)

    if backend_type == "pytorch":
        import torch
        tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            emb = model(tensor).squeeze(0).numpy()
        return emb.astype(np.float64)

    if backend_type == "tfhub":
        img_224 = np.array(Image.fromarray(
            (arr * arr.std() + arr.mean()).clip(0, 255).astype(np.uint8)
        ).resize((224, 224)), dtype=np.float32) / 255.0
        emb = model(tf.expand_dims(img_224, 0)).numpy().squeeze(0)
        return emb.astype(np.float64)

    arr_flat = arr.flatten().astype(np.float64)
    norm = np.linalg.norm(arr_flat)
    return arr_flat / norm if norm > 1e-10 else arr_flat


def extract_embedding_batch(images, model_backend=None) -> np.ndarray:
    if model_backend is None:
        model_backend = get_model()
    return np.stack([extract_embedding(img, model_backend) for img in images])