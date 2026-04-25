import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models

# Blend weights: shape/semantic vs colour.
# 0.65 shape + 0.35 colour means colour matters but isn't the only deciding factor.
CNN_WEIGHT = 0.70
COLOR_WEIGHT = 0.31

SIMILARITY_THRESHOLD = 0.73

_model = None
_device = None
_preprocess = None


def _get_model():
    global _model, _device, _preprocess
    if _model is not None:
        return _model, _device, _preprocess

    _device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    full_model = models.mobilenet_v3_small(weights=weights)
    full_model.eval()
    full_model.to(_device)
    _model = full_model.features
    _preprocess = weights.transforms()
    return _model, _device, _preprocess


def compute_embedding(image_path: str) -> list[float]:
    """576-dim L2-normalised CNN feature vector."""
    feature_extractor, device, preprocess = _get_model()
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(tensor)
        pooled = features.mean(dim=[2, 3])
        normed = F.normalize(pooled, dim=1)
        return normed.squeeze(0).cpu().tolist()


def compute_color_histogram(image_path: str) -> list[float]:
    """
    28-dim L2-normalised colour histogram (12 hue bins + 16 saturation bins).

    12 hue bins (30° each) keeps colour families together:
    red/pink/salmon all map to nearby bins even though their exact hue differs.
    Hue bins are weighted by saturation so achromatic pixels (white, grey)
    don't distort the hue signal.
    A Gaussian-style circular smoothing is applied to the hue histogram so
    that colours near a bin boundary (e.g. red at 0° vs pink at 355°) still
    score high similarity rather than landing in opposing bins.
    """
    with Image.open(image_path) as img:
        arr = np.array(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    s = np.where(cmax > 1e-6, delta / cmax, 0.0)

    h = np.zeros_like(r)
    mask = delta > 1e-6
    mr = mask & (cmax == r)
    mg = mask & (cmax == g)
    mb = mask & (cmax == b)
    h[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6
    h[mg] = (b[mg] - r[mg]) / delta[mg] + 2
    h[mb] = (r[mb] - g[mb]) / delta[mb] + 4
    h = h / 6.0  # 0–1

    h_flat = h.flatten()
    s_flat = s.flatten()

    # 12 hue bins weighted by saturation
    H_BINS = 12
    h_hist, _ = np.histogram(h_flat, bins=H_BINS, range=(0.0, 1.0), weights=s_flat)

    # Circular smoothing with a [0.25, 0.5, 0.25] kernel so neighbouring hue
    # bins (e.g. deep-red bin and pink bin) still contribute to each other.
    h_hist = h_hist.astype(np.float32)
    h_smooth = (
        0.5 * h_hist
        + 0.25 * np.roll(h_hist, 1)   # left neighbour (wraps: bin 0 ← bin 11)
        + 0.25 * np.roll(h_hist, -1)  # right neighbour
    )

    # 16 saturation bins (unweighted) capture colour intensity / vividness
    s_hist, _ = np.histogram(s_flat, bins=16, range=(0.0, 1.0))

    hist = np.concatenate([h_smooth, s_hist.astype(np.float32)])
    norm = np.linalg.norm(hist)
    if norm > 1e-6:
        hist /= norm
    return hist.tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.dot(va, vb))


def blended_similarity(
    emb_a: list[float], hist_a: list[float],
    emb_b: list[float], hist_b: list[float],
) -> float:
    cnn_sim   = _cosine(emb_a, emb_b)
    color_sim = _cosine(hist_a, hist_b)
    return CNN_WEIGHT * cnn_sim + COLOR_WEIGHT * color_sim


def find_or_create_collection(db, embedding: list[float], color_histogram: list[float]):
    from models import Photo, Collection

    photos = db.query(Photo).filter(
        Photo.embedding.isnot(None),
        Photo.color_histogram.isnot(None),
    ).all()

    best_sim = SIMILARITY_THRESHOLD - 0.001
    best_collection_id = None

    for photo in photos:
        sim = blended_similarity(
            embedding, color_histogram,
            json.loads(photo.embedding),
            json.loads(photo.color_histogram),
        )
        if sim > best_sim:
            best_sim = sim
            best_collection_id = photo.collection_id

    if best_collection_id is not None:
        return best_collection_id, False

    count = db.query(Collection).count()
    new_collection = Collection(name=f"Collection {count + 1}")
    db.add(new_collection)
    db.flush()
    return new_collection.id, True
