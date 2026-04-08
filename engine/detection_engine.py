from __future__ import annotations

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from detectors import (
    ApneaDetector,
    ArrhythmiaDetector,
    HypoxiaDetector,
    SepsisDetector,
)


@dataclass
class AlarmState:
    state: str = "WARMING_UP"
    positive_streak: int = 0
    recovery_streak: int = 0


class AlarmController:
    def __init__(self, suspect_steps=3, alarm_steps=5, recovery_steps=3):
        self.suspect_steps = suspect_steps
        self.alarm_steps = alarm_steps
        self.recovery_steps = recovery_steps

    def update(self, alarm_state, prob, threshold, activation_allowed=True):
        if prob is None:
            alarm_state.state = "WARMING_UP"
            alarm_state.positive_streak = 0
            alarm_state.recovery_streak = 0
            return alarm_state.state

        if activation_allowed and prob >= threshold:
            alarm_state.positive_streak += 1
            alarm_state.recovery_streak = 0

            if alarm_state.positive_streak >= self.alarm_steps:
                alarm_state.state = "ALARM"
            elif alarm_state.positive_streak >= self.suspect_steps:
                alarm_state.state = "SUSPECT"
            else:
                alarm_state.state = "NORMAL"
            return alarm_state.state

        alarm_state.positive_streak = 0

        if prob < 0.3:
            alarm_state.recovery_streak += 1
        else:
            alarm_state.recovery_streak = 0

        if alarm_state.recovery_streak >= 6:
            alarm_state.state = "NORMAL"
        elif alarm_state.recovery_streak >= 3:
            alarm_state.state = "RECOVERING"
        else:
            alarm_state.state = "NORMAL"

        return alarm_state.state


class DetectionEngine:
    def __init__(self):
        self.entry_threshold = 0.55
        self.exit_threshold = 0.45
        self.separation_margin = 0.15
        self.switch_margin = 0.12

        self.hypoxia_lock_threshold = 0.72
        self.arrhythmia_spike_threshold = 0.7

        self.detectors = {
            "apnea": ApneaDetector(),
            "sepsis": SepsisDetector(),
            "hypoxia": HypoxiaDetector(),
            "arrhythmia": ArrhythmiaDetector(),
        }

        self.thresholds = {
            "apnea": 0.55,
            "sepsis": 0.60,
            "hypoxia": 0.58,
            "arrhythmia": 0.55,
        }

        self.alarm_states = {name: AlarmState() for name in self.detectors}
        self.alarm_controller = AlarmController()

        self.executor = ThreadPoolExecutor(max_workers=len(self.detectors))

        self.prev_dominant = None
        self.sample_index = 0

    def _dampen(self, p):
        if p is None:
            return None
        if p < 0.4:
            p *= 0.4
        return float(np.clip(p, 0.01, 0.95))

    def _coordinate(self, probs):
        adjusted = {k: self._dampen(v) for k, v in probs.items()}

        # sepsis shaping
        if adjusted["sepsis"] is not None:
            if self.sample_index < 300:
                adjusted["sepsis"] *= 0.2
            else:
                adjusted["sepsis"] *= 1.2

        # cross inhibition
        if adjusted["hypoxia"] is not None and adjusted["hypoxia"] > 0.6:
            adjusted["apnea"] *= 0.05

        if adjusted["apnea"] is not None and adjusted["apnea"] > 0.65:
            if adjusted["hypoxia"] is not None and adjusted["hypoxia"] < 0.45:
                adjusted["hypoxia"] *= 0.9

        if adjusted["hypoxia"] is not None and adjusted["hypoxia"] > 0.6:
            if adjusted["arrhythmia"] is not None:
                adjusted["arrhythmia"] *= 0.6

        if any(
            adjusted[k] is not None and adjusted[k] > 0.75
            for k in ["apnea", "hypoxia", "arrhythmia"]
        ):
            if adjusted["sepsis"] is not None:
                adjusted["sepsis"] *= 0.7

        for k in adjusted:
            if adjusted[k] is not None:
                adjusted[k] = float(np.clip(adjusted[k], 0.01, 0.95))

        return adjusted

    def process_sample(self, sample):
        futures = {
            k: self.executor.submit(det.update, sample)
            for k, det in self.detectors.items()
        }

        raw = {k: f.result() for k, f in futures.items()}
        probs = self._coordinate(raw)

        available = {k: v for k, v in probs.items() if v is not None}

        candidate, candidate_prob = None, None
        second_prob = None

        if available:
            ranked = sorted(available.items(), key=lambda x: x[1], reverse=True)
            candidate, candidate_prob = ranked[0]
            second_prob = ranked[1][1] if len(ranked) > 1 else None

            if candidate_prob is not None and candidate_prob < self.entry_threshold:
                candidate = None
                candidate_prob = None

            if (
                candidate is not None
                and candidate_prob is not None
                and second_prob is not None
            ):
                if (candidate_prob - second_prob) < self.separation_margin:
                    candidate = None
                    candidate_prob = None

        prev = self.prev_dominant
        prev_prob = probs.get(prev) if prev else None
        hypoxia_prob = probs.get("hypoxia")
        sepsis_prob = probs.get("sepsis")

        # dominance logic
        if prev is None:
            if candidate == "hypoxia" and candidate_prob is not None and candidate_prob >= self.hypoxia_lock_threshold:
                dominant = "hypoxia"
            else:
                dominant = candidate
        elif candidate == prev:
            dominant = prev
        elif candidate is None:
            if prev_prob is None or prev_prob < self.exit_threshold:
                if prev_prob is not None and prev_prob > (self.exit_threshold - 0.05):
                    dominant = prev
                else:
                    dominant = None
            else:
                dominant = prev
        else:
            if candidate == "hypoxia" and candidate_prob is not None and candidate_prob >= self.hypoxia_lock_threshold:
                dominant = "hypoxia"
            elif prev == "hypoxia" and prev_prob is not None:
                if candidate_prob is None:
                    dominant = prev
                elif candidate_prob <= prev_prob + self.switch_margin:
                    dominant = prev
                else:
                    dominant = candidate
            elif candidate == "apnea" and prev == "hypoxia":
                if prev_prob is None or prev_prob < self.exit_threshold:
                    dominant = candidate if candidate_prob is not None else prev
                else:
                    dominant = prev
            elif candidate == "arrhythmia":
                if (
                    candidate_prob is not None
                    and candidate_prob >= self.arrhythmia_spike_threshold
                    and (prev_prob is None or candidate_prob > prev_prob + self.switch_margin)
                ):
                    dominant = candidate
                else:
                    dominant = prev
            elif candidate == "sepsis":
                if (
                    self.sample_index >= 300
                    and sepsis_prob is not None
                    and sepsis_prob >= self.thresholds["sepsis"] + 0.05
                    and (prev_prob is None or sepsis_prob > prev_prob + self.switch_margin)
                ):
                    dominant = candidate
                else:
                    dominant = prev
            elif prev_prob is None or (
                candidate_prob is not None and candidate_prob > prev_prob + self.switch_margin
            ):
                dominant = candidate
            else:
                dominant = prev

        self.prev_dominant = dominant

        results = {}
        for name, prob in probs.items():
            allow = name == dominant
            state = self.alarm_controller.update(
                self.alarm_states[name], prob, self.thresholds[name], allow
            )
            results[name] = {"probability": prob, "state": state}

        if dominant == "hypoxia" and results["hypoxia"]["probability"] is not None:
            results["hypoxia"]["state"] = "ALARM"

        results["dominant"] = {
            "probability": probs.get(dominant) if dominant else None,
            "state": dominant or "NONE",
        }

        self.sample_index += 1
        return results

    def shutdown(self):
        self.executor.shutdown(wait=True)


def _apply_event(sample, t, start, end, delta_fn):
    if start <= t < end:
        sample += delta_fn(t - start, end - start)


def simulate_realtime_sample(t):
    base = np.array(
        [
            np.random.normal(74, 2.8),
            np.random.normal(15.8, 1.3),
            np.random.normal(97.2, 0.5),
            np.random.beta(2.0, 5.0),
        ],
        dtype=np.float32,
    )

    _apply_event(
        base,
        t,
        start=90,
        end=120,
        delta_fn=lambda i, n: np.array(
            [
                np.random.normal(2.0, 1.2),
                -np.interp(i, [0, n - 1], [2.5, 8.0]),
                -np.interp(i, [0, n - 1], [1.0, 4.2]),
                np.random.uniform(-0.05, 0.1),
            ],
            dtype=np.float32,
        ),
    )

    _apply_event(
        base,
        t,
        start=160,
        end=205,
        delta_fn=lambda i, n: np.array(
            [
                np.interp(i, [0, n - 1], [0.5, 7.0]),
                np.interp(i, [0, n - 1], [1.0, 6.5]),
                -np.interp(i, [0, n - 1], [2.0, 12.0]),
                np.random.uniform(0.05, 0.25),
            ],
            dtype=np.float32,
        ),
    )

    _apply_event(
        base,
        t,
        start=230,
        end=250,
        delta_fn=lambda i, n: np.array(
            [
                np.random.uniform(32, 62) if i % 2 == 0 else np.random.uniform(-28, -18),
                np.random.uniform(-1.5, 3.0),
                -np.interp(i, [0, n - 1], [0.8, 2.4]),
                np.random.uniform(0.05, 0.35),
            ],
            dtype=np.float32,
        ),
    )

    _apply_event(
        base,
        t,
        start=300,
        end=420,
        delta_fn=lambda i, n: np.array(
            [
                np.interp(i, [0, n - 1], [4.0, 34.0]),
                np.interp(i, [0, n - 1], [2.0, 14.0]),
                -np.interp(i, [0, n - 1], [1.0, 10.5]),
                np.random.uniform(-0.08, 0.2),
            ],
            dtype=np.float32,
        ),
    )

    base[0] = float(np.clip(base[0], 35.0, 180.0))
    base[1] = float(np.clip(base[1], 4.0, 45.0))
    base[2] = float(np.clip(base[2], 70.0, 100.0))
    base[3] = float(np.clip(base[3], 0.02, 1.0))

    return base.tolist()


def _severity_rank(state):
    order = {
        "WARMING_UP": 0,
        "NORMAL": 1,
        "RECOVERING": 2,
        "SUSPECT": 3,
        "ALARM": 4,
    }
    return order.get(state, 0)


def _global_state(results):
    states = [
        entry["state"]
        for name, entry in results.items()
        if name != "dominant"
    ]
    return max(states, key=_severity_rank)


if __name__ == "__main__":
    engine = DetectionEngine()
    print("Multi-disease detection engine started\n")
    print("Columns: time | HR RR SpO2 Move | apnea | sepsis | hypoxia | arrhythmia | GLOBAL")

    t = 0
    try:
        while True:
            sample = simulate_realtime_sample(t)
            out = engine.process_sample(sample)
            global_state = _global_state(out)

            def _fmt(name):
                prob = out[name]["probability"]
                state = out[name]["state"]
                ptxt = "----" if prob is None else f"{prob:.3f}"
                return f"{name}:{ptxt}/{state}"

            dominant = out["dominant"]["state"]

            print(
                f"t={t:04d}s | HR={sample[0]:6.1f} RR={sample[1]:5.1f} SpO2={sample[2]:6.1f} "
                f"Move={sample[3]:.2f} | "
                f"{_fmt('apnea')} | {_fmt('sepsis')} | {_fmt('hypoxia')} | {_fmt('arrhythmia')} | "
                f"GLOBAL={global_state} | DOMINANT={dominant}"
            )

            time.sleep(1)
            t += 1
    except KeyboardInterrupt:
        print("\nStopping detection engine...")
    finally:
        engine.shutdown()
