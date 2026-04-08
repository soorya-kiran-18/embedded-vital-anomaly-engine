from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np


ArrayLikePath = Union[str, Path]


@dataclass(frozen=True)
class NormalizationStats:
    means: np.ndarray  # shape (3,)
    stds: np.ndarray   # shape (3,)


def compute_normalization_stats(x: np.ndarray) -> NormalizationStats:
    """
    Matches the existing apnea logic exactly:
    - Normalize only HR, RR, SpO2 (feature indices 0..2)
    - Movement (index 3) remains unchanged
    - Mean/std are computed over all windows and all timesteps
    """
    if x.ndim != 3 or x.shape[-1] < 4:
        raise ValueError("Expected x with shape (N, T, 4).")

    means = np.array([x[:, :, idx].mean() for idx in range(3)], dtype=np.float32)
    stds = np.array([x[:, :, idx].std() for idx in range(3)], dtype=np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)

    return NormalizationStats(means=means, stds=stds)


def apply_normalization(x: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    x_norm = x.astype(np.float32, copy=True)
    for idx in range(3):
        x_norm[:, :, idx] = (x_norm[:, :, idx] - stats.means[idx]) / stats.stds[idx]
    return x_norm


def apply_normalization_single_window(window: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    if window.ndim != 2 or window.shape[-1] < 4:
        raise ValueError("Expected window with shape (T, 4).")

    out = window.astype(np.float32, copy=True)
    for idx in range(3):
        out[:, idx] = (out[:, idx] - stats.means[idx]) / stats.stds[idx]
    return out


def save_stats(path: ArrayLikePath, stats: NormalizationStats) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, means=stats.means, stds=stats.stds)


def load_stats(path: ArrayLikePath) -> NormalizationStats:
    loaded = np.load(path)
    return NormalizationStats(
        means=loaded["means"].astype(np.float32),
        stds=loaded["stds"].astype(np.float32),
    )


def load_or_compute_stats(
    stats_path: ArrayLikePath,
    dataset_fallback_path: ArrayLikePath,
) -> NormalizationStats:
    stats_path = Path(stats_path)
    dataset_fallback_path = Path(dataset_fallback_path)

    if stats_path.exists():
        return load_stats(stats_path)

    if not dataset_fallback_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found at '{stats_path}', and fallback dataset "
            f"'{dataset_fallback_path}' does not exist."
        )

    x = np.load(dataset_fallback_path)
    stats = compute_normalization_stats(x)
    save_stats(stats_path, stats)
    return stats
