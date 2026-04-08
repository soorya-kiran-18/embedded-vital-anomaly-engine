from __future__ import annotations

import numpy as np


WINDOW_SIZE = 60  # compressed long-term trend
NUM_FEATURES = 4


def _event_bounds() -> tuple[int, int, int]:
    event_len = np.random.randint(int(0.35 * WINDOW_SIZE), int(0.70 * WINDOW_SIZE) + 1)
    event_start = np.random.randint(4, max(5, WINDOW_SIZE - event_len - 2))
    event_end = event_start + event_len
    return event_start, event_end, WINDOW_SIZE - event_end


def _gradual_trend(length: int, peak: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length)
    exponent = np.random.uniform(1.2, 2.0)
    trend = (x ** exponent) * peak
    drift = np.cumsum(np.random.normal(0.0, peak * 0.010, length))
    undulation = np.sin(np.linspace(0.0, np.random.uniform(1.5, 3.0) * np.pi, length)) * peak * 0.05
    out = trend + drift + undulation + np.random.normal(0.0, peak * 0.04, length)
    return out.astype(np.float32)


def _movement_pattern(hr: np.ndarray, rr: np.ndarray, event_slice: slice, trend_strength: float) -> np.ndarray:
    length = hr.shape[0]
    base = np.random.beta(2.0, 5.0, size=length).astype(np.float32)
    hr_slope = np.abs(np.gradient(hr))
    rr_slope = np.abs(np.gradient(rr))
    hr_slope = hr_slope / (np.percentile(hr_slope, 90) + 1e-6)
    rr_slope = rr_slope / (np.percentile(rr_slope, 90) + 1e-6)

    move = base + 0.08 * hr_slope.astype(np.float32) + 0.08 * rr_slope.astype(np.float32)

    event_len = event_slice.stop - event_slice.start
    move[event_slice] += np.random.uniform(-0.02, 0.06, event_len).astype(np.float32) * trend_strength

    bursts = max(2, length // 14)
    burst_idx = np.random.choice(np.arange(length), size=bursts, replace=False)
    move[burst_idx] += np.random.uniform(0.06, 0.24, size=bursts).astype(np.float32)

    return np.clip(move, 0.03, 1.0)


def _normal_window() -> np.ndarray:
    event_start, event_end, recovery_len = _event_bounds()
    event_len = event_end - event_start

    hr_base = np.random.uniform(62, 90)
    rr_base = np.random.uniform(11, 19)
    spo2_base = np.random.uniform(95.2, 99.2)

    hr = hr_base + np.random.normal(0.0, np.random.uniform(2.0, 4.4), WINDOW_SIZE)
    rr = rr_base + np.random.normal(0.0, np.random.uniform(1.0, 2.2), WINDOW_SIZE)
    spo2 = spo2_base + np.random.normal(0.0, np.random.uniform(0.5, 1.1), WINDOW_SIZE)

    # Hard negatives: mild fatigue-like trend with limited amplitude.
    hr[event_start:event_end] += _gradual_trend(event_len, np.random.uniform(2.0, 8.0))
    rr[event_start:event_end] += _gradual_trend(event_len, np.random.uniform(0.8, 3.0))
    spo2[event_start:event_end] -= np.clip(_gradual_trend(event_len, np.random.uniform(0.5, 2.2)), 0.0, None)

    if recovery_len > 0:
        hr[event_end:] -= np.linspace(0.0, np.random.uniform(0.5, 2.8), recovery_len)
        rr[event_end:] -= np.linspace(0.0, np.random.uniform(0.3, 1.4), recovery_len)
        spo2[event_end:] += np.linspace(0.0, np.random.uniform(0.2, 1.0), recovery_len)

    movement = _movement_pattern(hr, rr, slice(event_start, event_end), trend_strength=0.8)
    return np.column_stack((hr, rr, spo2, movement)).astype(np.float32)


def _sepsis_window() -> np.ndarray:
    event_start, event_end, recovery_len = _event_bounds()
    event_len = event_end - event_start

    hr_base = np.random.uniform(64, 92)
    rr_base = np.random.uniform(12, 20)
    spo2_base = np.random.uniform(94.8, 98.8)

    hr = hr_base + np.random.normal(0.0, np.random.uniform(2.3, 5.0), WINDOW_SIZE)
    rr = rr_base + np.random.normal(0.0, np.random.uniform(1.1, 2.4), WINDOW_SIZE)
    spo2 = spo2_base + np.random.normal(0.0, np.random.uniform(0.6, 1.2), WINDOW_SIZE)

    severity = np.random.choice(["subtle", "mild", "moderate", "severe"], p=[0.22, 0.34, 0.30, 0.14])
    if severity == "subtle":
        hr_rise = np.random.uniform(6.0, 12.0)
        rr_rise = np.random.uniform(2.0, 5.0)
        spo2_drop = np.random.uniform(1.0, 2.8)
        trend_strength = 0.8
    elif severity == "mild":
        hr_rise = np.random.uniform(10.0, 18.0)
        rr_rise = np.random.uniform(4.0, 7.5)
        spo2_drop = np.random.uniform(2.0, 4.0)
        trend_strength = 1.0
    elif severity == "moderate":
        hr_rise = np.random.uniform(16.0, 26.0)
        rr_rise = np.random.uniform(6.5, 11.0)
        spo2_drop = np.random.uniform(3.5, 6.2)
        trend_strength = 1.2
    else:
        hr_rise = np.random.uniform(24.0, 34.0)
        rr_rise = np.random.uniform(9.0, 15.0)
        spo2_drop = np.random.uniform(5.5, 8.8)
        trend_strength = 1.4

    # Core sepsis signature: gradual long-trend deterioration.
    hr[event_start:event_end] += _gradual_trend(event_len, hr_rise)
    rr[event_start:event_end] += _gradual_trend(event_len, rr_rise)
    spo2[event_start:event_end] -= np.clip(_gradual_trend(event_len, spo2_drop), 0.0, None)

    if recovery_len > 0:
        # Sepsis recovery is slow/incomplete in-window.
        rec = np.random.uniform(0.20, 0.55)
        hr[event_end:] -= np.linspace(0.1, hr_rise * rec, recovery_len)
        rr[event_end:] -= np.linspace(0.1, rr_rise * rec, recovery_len)
        spo2[event_end:] += np.linspace(0.1, spo2_drop * rec, recovery_len)

    movement = _movement_pattern(hr, rr, slice(event_start, event_end), trend_strength=trend_strength)
    return np.column_stack((hr, rr, spo2, movement)).astype(np.float32)


def generate_sepsis_dataset(
    n_normal: int = 2600,
    n_sepsis: int = 2600,
    seed: int = 44,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = []
    y = []

    for _ in range(n_normal):
        x.append(_normal_window())
        y.append(0)
    for _ in range(n_sepsis):
        x.append(_sepsis_window())
        y.append(1)

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


if __name__ == "__main__":
    X, y = generate_sepsis_dataset()
    np.save("models/sepsis_X.npy", X)
    np.save("models/sepsis_y.npy", y)
    print("Sepsis dataset generated")
    print("X:", X.shape, "y:", y.shape)
