from __future__ import annotations

import numpy as np


WINDOW_SIZE = 30  # 20-40s requirement satisfied
NUM_FEATURES = 4


def _event_bounds() -> tuple[int, int, int]:
    event_len = np.random.randint(int(0.30 * WINDOW_SIZE), int(0.70 * WINDOW_SIZE) + 1)
    event_start = np.random.randint(3, max(4, WINDOW_SIZE - event_len - 2))
    event_end = event_start + event_len
    return event_start, event_end, WINDOW_SIZE - event_end


def _event_profile(length: int, peak: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length)
    rise = 1.0 / (1.0 + np.exp(-9.0 * (x - np.random.uniform(0.18, 0.33))))
    fall = 1.0 / (1.0 + np.exp(8.0 * (x - np.random.uniform(0.65, 0.88))))
    profile = rise * fall
    profile = profile / (np.max(profile) + 1e-6)
    profile = profile * peak
    wobble = np.cumsum(np.random.normal(0.0, peak * 0.015, length))
    profile = profile + wobble + np.random.normal(0.0, peak * 0.10, length)
    return np.clip(profile, 0.0, None).astype(np.float32)


def _movement_pattern(length: int, hr: np.ndarray, rr: np.ndarray, event_slice: slice, event_strength: float) -> np.ndarray:
    base = np.random.beta(2.4, 4.8, size=length).astype(np.float32)
    hr_dyn = np.abs(np.diff(hr, prepend=hr[0]))
    rr_dyn = np.abs(np.diff(rr, prepend=rr[0]))
    hr_dyn = hr_dyn / (np.percentile(hr_dyn, 90) + 1e-6)
    rr_dyn = rr_dyn / (np.percentile(rr_dyn, 90) + 1e-6)
    move = base + 0.10 * hr_dyn.astype(np.float32) + 0.10 * rr_dyn.astype(np.float32)

    event_len = event_slice.stop - event_slice.start
    move[event_slice] += np.random.uniform(0.05, 0.22, event_len).astype(np.float32) * event_strength

    burst_count = max(2, length // 10)
    burst_idx = np.random.choice(np.arange(length), size=burst_count, replace=False)
    move[burst_idx] += np.random.uniform(0.08, 0.32, size=burst_count).astype(np.float32)

    return np.clip(move, 0.03, 1.0)


def _normal_window() -> np.ndarray:
    event_start, event_end, recovery_len = _event_bounds()
    event_len = event_end - event_start

    hr_base = np.random.uniform(62, 88)
    rr_base = np.random.uniform(12, 19)
    spo2_base = np.random.uniform(95.2, 99.0)

    hr = hr_base + np.random.normal(0.0, np.random.uniform(2.0, 4.2), WINDOW_SIZE)
    rr = rr_base + np.random.normal(0.0, np.random.uniform(1.0, 2.0), WINDOW_SIZE)
    spo2 = spo2_base + np.random.normal(0.0, np.random.uniform(0.5, 1.0), WINDOW_SIZE)

    # Hard negatives: mild oxygen dip + mild respiratory compensation.
    spo2[event_start:event_end] -= _event_profile(event_len, np.random.uniform(0.6, 2.6))
    rr[event_start:event_end] += _event_profile(event_len, np.random.uniform(0.2, 1.8))
    hr[event_start:event_end] += _event_profile(event_len, np.random.uniform(0.0, 1.8))

    if recovery_len > 0:
        rec_strength = np.random.uniform(0.25, 0.70)
        spo2[event_end:] += np.linspace(0.0, rec_strength * 1.4, recovery_len)
        rr[event_end:] -= np.linspace(0.0, rec_strength * 1.0, recovery_len)
        hr[event_end:] -= np.linspace(0.0, rec_strength * 1.2, recovery_len)

    movement = _movement_pattern(WINDOW_SIZE, hr, rr, slice(event_start, event_end), event_strength=0.6)
    return np.column_stack((hr, rr, spo2, movement)).astype(np.float32)


def _hypoxia_window() -> np.ndarray:
    event_start, event_end, recovery_len = _event_bounds()
    event_len = event_end - event_start

    hr_base = np.random.uniform(64, 92)
    rr_base = np.random.uniform(12, 21)
    spo2_base = np.random.uniform(94.8, 98.9)

    hr = hr_base + np.random.normal(0.0, np.random.uniform(2.3, 5.0), WINDOW_SIZE)
    rr = rr_base + np.random.normal(0.0, np.random.uniform(1.1, 2.4), WINDOW_SIZE)
    spo2 = spo2_base + np.random.normal(0.0, np.random.uniform(0.6, 1.2), WINDOW_SIZE)

    severity = np.random.choice(["subtle", "mild", "moderate", "severe"], p=[0.22, 0.33, 0.30, 0.15])
    if severity == "subtle":
        spo2_drop = np.random.uniform(2.3, 4.2)
        rr_rise = np.random.uniform(1.2, 3.2)
        hr_rise = np.random.uniform(0.8, 2.8)
        ev_strength = 0.8
    elif severity == "mild":
        spo2_drop = np.random.uniform(3.8, 6.0)
        rr_rise = np.random.uniform(2.0, 4.8)
        hr_rise = np.random.uniform(1.5, 4.5)
        ev_strength = 1.0
    elif severity == "moderate":
        spo2_drop = np.random.uniform(5.5, 8.2)
        rr_rise = np.random.uniform(3.5, 6.8)
        hr_rise = np.random.uniform(3.0, 6.5)
        ev_strength = 1.2
    else:
        spo2_drop = np.random.uniform(7.2, 10.2)
        rr_rise = np.random.uniform(5.5, 8.8)
        hr_rise = np.random.uniform(5.0, 9.0)
        ev_strength = 1.4

    spo2[event_start:event_end] -= _event_profile(event_len, spo2_drop)
    rr[event_start:event_end] += _event_profile(event_len, rr_rise)
    hr[event_start:event_end] += _event_profile(event_len, hr_rise)

    if recovery_len > 0:
        rec = np.random.uniform(0.35, 0.80)
        spo2[event_end:] += np.linspace(0.2, spo2_drop * rec, recovery_len)
        rr[event_end:] -= np.linspace(0.1, rr_rise * rec, recovery_len)
        hr[event_end:] -= np.linspace(0.1, hr_rise * rec, recovery_len)

    movement = _movement_pattern(WINDOW_SIZE, hr, rr, slice(event_start, event_end), event_strength=ev_strength)
    return np.column_stack((hr, rr, spo2, movement)).astype(np.float32)


def generate_hypoxia_dataset(
    n_normal: int = 2800,
    n_hypoxia: int = 2800,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = []
    y = []

    for _ in range(n_normal):
        x.append(_normal_window())
        y.append(0)
    for _ in range(n_hypoxia):
        x.append(_hypoxia_window())
        y.append(1)

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


if __name__ == "__main__":
    X, y = generate_hypoxia_dataset()
    np.save("models/hypoxia_X.npy", X)
    np.save("models/hypoxia_y.npy", y)
    print("Hypoxia dataset generated")
    print("X:", X.shape, "y:", y.shape)
