from __future__ import annotations

import numpy as np
import pandas as pd


def as_2d_array(data, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D numeric array.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def feature_names(data, prefix: str) -> list[str]:
    if isinstance(data, pd.DataFrame):
        return [str(col) for col in data.columns]
    arr = as_2d_array(data, prefix)
    width = len(str(arr.shape[1]))
    return [f"{prefix}{idx:0{width}d}" for idx in range(1, arr.shape[1] + 1)]


def sample_index(data, n_rows: int) -> list:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return list(data.index)
    return list(range(n_rows))


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def mixomics_keep_threshold(weights: np.ndarray, keep: int) -> np.ndarray:
    """Apply mixOmics-style cardinality soft thresholding."""
    weights = np.asarray(weights, dtype=float)
    if keep < 1:
        raise ValueError("keep values must be at least 1.")
    if keep >= weights.size:
        return weights.copy()

    n_drop = weights.size - keep
    abs_weights = np.abs(weights)
    order = np.argsort(abs_weights, kind="mergesort")
    drop_idx = order[:n_drop]
    keep_idx = order[n_drop:]
    threshold = abs_weights[drop_idx].max()

    out = np.zeros_like(weights)
    out[keep_idx] = np.sign(weights[keep_idx]) * (abs_weights[keep_idx] - threshold)
    return out


def default_keep_options(n_features: int) -> list[int]:
    if n_features <= 10:
        return list(range(1, n_features + 1))

    step = int(np.ceil(n_features / 10))
    options = list(range(0, n_features + 1, step))
    options[0] = 1
    if options[-1] != n_features:
        options.append(n_features)
    return sorted(set(options))
