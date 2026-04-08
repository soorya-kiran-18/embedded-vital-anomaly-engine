from __future__ import annotations

import numpy as np


WINDOW_SIZE = 16  # 10-20s requirement satisfied
NUM_FEATURES = 4


def _event_bounds() -> tuple[int, int, int]:
    event_len = np.random.randint(int(0.30 * WINDOW_SIZE), int(0.70 * WINDOW_SIZE) + 1)
    event_start = np.random.randint(2, max(3, WINDOW_SIZE - event_len - 1))
    event_end = event_start + event_len
    return event_start, event_end, WINDOW_SIZE - event_end


def _profile(length: int, peak: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length)
    rise = 1.0 / (1.0 + np.exp(-10.0 * (x - np.random.uniform(0.18, 0.32))))
    fall = 1.0 / (1.0 + np.exp(8.0 * (x - np.random.uniform(0.62, 0.86))))
    env = rise * fall
    env = env / (np.max(env) + 1e-6)
    env = env * peak
    env += np.random.normal(0.0, peak * 0.12, length)
    return env.astype(np.float32)


def _movement_pattern(hr: np.ndarray, rr: np.ndarray, event_slice: slice, burst_level: float) -> np.ndarray:
    length = hr.shape[0]
    base = np.random.beta(2.1, 4.9, size=length).astype(np.float32)
    hr_dyn = np.abs(np.diff(hr, prepend=hr[0]))
    rr_dyn = np.abs(np.diff(rr, prepend=rr[0]))
    hr_dyn = hr_dyn / (np.percentile(hr_dyn, 90) + 1e-6)
    rr_dyn = rr_dyn / (np.percentile(rr_dyn, 90) + 1e-6)
    move = base + 0.14 * hr_dyn.astype(np.float32) + 0.04 * rr_dyn.astype(np.float32)

    event_len = event_slice.stop - event_slice.start
    move[event_slice] += np.random.uniform(-0.02, 0.08, event_len).astype(np.float32)

    burst_count = max(1, length // 4)
    burst_idx = np.random.choice(np.arange(length), size=burst_count, replace=False)
    move[burst_idx] += np.random.uniform(0.12, 0.45, size=burst_count).astype(np.float32) * burst_level

    return np.clip(move, 0.03, 1.0)


def _normal_window() -> np.ndarray:
    event_start, event_end, recovery_len = _event_bounds()
    event_len = event_end - event_start

    hr_base = np.random.uniform(60, 88)
    rr_base = np.random.uniform(12, 19)
    spo2_base = np.random.uniform(95.0, 99.0)

    hr = hr_base + np.random.normal(0.0, np.random.uniform(2.0, 4.5), WINDOW_SIZE)
    rr = rr_base + np.random.normal(0.0, np.random.uniform(1.0, 2.2), WINDOW_SIZE)
    spo2 = spo2_base + np.random.normal(0.0, np.random.uniform(0.5, 1.0), WINDOW_SIZE)

    # Borderline benign rhythm variability.
    hr[event_start:event_end] += _profile(event_len, np.random.uniform(3.0, 10.0))
    hr[event_start:event_end] += np.random.normal(0.0, np.random.uniform(2.0, 5.0), event_len)
    rr[event_start:event_end] += _profile(event_len, np.random.uniform(0.0, 1.2))
    spo2[event_start:event_end] -= np.clip(_profile(event_len, np.random.uniform(0.0, 0.8)), 0.0, None)

    if recovery_len > 0:
        hr[event_end:] -= np.linspace(0.0, np.random.uniform(0.5, 2.0), recovery_len)
        rr[event_end:] -= np.linspace(0.0, np.random.uniform(0.1, 0.6), recovery_len)
        spo2[event_end:] += np.linspace(0.0, np.random.uniform(0.1, 0.8), recovery_len)

    movement = _movement_pattern(hr, rr, slice(event_start, event_end), burst_level=0.8)
    return np.column_stack((hr, rr, spo2, movement)).astype(np.float32)


def _arrhythmia_window() -> np.ndarray:
    event_start, event_end, recovery_len = _event_bounds()
    event_len = event_end - event_start

    hr_base = np.random.uniform(60, 92)
    rr_base = np.random.uniform(12, 20)
    spo2_base = np.random.uniform(94.6, 98.8)

    hr = hr_base + np.random.normal(0.0, np.random.uniform(2.3, 5.0), WINDOW_SIZE)
    rr = rr_base + np.random.normal(0.0, np.random.uniform(1.0, 2.3), WINDOW_SIZE)
    spo2 = spo2_base + np.random.normal(0.0, np.random.uniform(0.5, 1.1), WINDOW_SIZE)

    severity = np.random.choice(["subtle", "mild", "moderate", "severe"], p=[0.25, 0.33, 0.28, 0.14])
    if severity == "subtle":
        hr_amp = np.random.uniform(10.0, 18.0)
        rr_shift = np.random.uniform(0.0, 0.8)
        spo2_drop = np.random.uniform(0.2, 1.0)
        burst_level = 0.9
    elif severity == "mild":
        hr_amp = np.random.uniform(16.0, 26.0)
        rr_shift = np.random.uniform(0.2, 1.2)
        spo2_drop = np.random.uniform(0.5, 1.6)
        burst_level = 1.0
    elif severity == "moderate":
        hr_amp = np.random.uniform(24.0, 38.0)
        rr_shift = np.random.uniform(0.4, 1.8)
        spo2_drop = np.random.uniform(1.0, 2.4)
        burst_level = 1.2
    else:
        hr_amp = np.random.uniform(34.0, 52.0)
        rr_shift = np.random.uniform(0.8, 2.5)
        spo2_drop = np.random.uniform(1.8, 3.2)
        burst_level = 1.4

    # Primary arrhythmia signature: irregular HR spikes and drops.
    env = _profile(event_len, hr_amp)
    direction = np.random.choice(np.array([-1.0, 1.0], dtype=np.float32), size=event_len, p=[0.45, 0.55])
    impulses = np.zeros(event_len, dtype=np.float32)
    impulse_idx = np.random.choice(np.arange(event_len), size=max(2, event_len // 2), replace=False)
    impulses[impulse_idx] = np.random.uniform(0.4, 1.0, size=impulse_idx.shape[0]).astype(np.float32)
    hr[event_start:event_end] += env * direction * (0.5 + impulses)
    hr[event_start:event_end] += np.random.normal(0.0, np.random.uniform(4.0, 8.5), event_len)

    # Minimal RR influence and mild oxygen effect.
    rr[event_start:event_end] += _profile(event_len, rr_shift) * np.random.uniform(-0.4, 0.6)
    spo2[event_start:event_end] -= np.clip(_profile(event_len, spo2_drop), 0.0, None)

    if recovery_len > 0:
        rec = np.random.uniform(0.30, 0.80)
        hr[event_end:] -= np.linspace(0.2, hr_amp * rec * 0.30, recovery_len)
        rr[event_end:] -= np.linspace(0.0, rr_shift * rec * 0.6, recovery_len)
        spo2[event_end:] += np.linspace(0.0, spo2_drop * rec, recovery_len)

    movement = _movement_pattern(hr, rr, slice(event_start, event_end), burst_level=burst_level)
    return np.column_stack((hr, rr, spo2, movement)).astype(np.float32)


def generate_arrhythmia_dataset(
    n_normal: int = 3200,
    n_arrhythmia: int = 3200,
    seed: int = 43,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = []
    y = []

    for _ in range(n_normal):
        x.append(_normal_window())
        y.append(0)
    for _ in range(n_arrhythmia):
        x.append(_arrhythmia_window())
        y.append(1)

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


if __name__ == "__main__":
    X, y = generate_arrhythmia_dataset()
    np.save("models/arrhythmia_X.npy", X)
    np.save("models/arrhythmia_y.npy", y)
    print("Arrhythmia dataset generated")
    print("X:", X.shape, "y:", y.shape)
