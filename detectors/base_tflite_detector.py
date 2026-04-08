from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Optional, Sequence

import numpy as np
import tensorflow as tf

from preprocessing import apply_normalization_single_window, load_or_compute_stats


class BaseTFLiteDetector:
    """Common buffered TFLite detector with shared normalization."""

    def __init__(
        self,
        model_path: str,
        stats_path: str,
        dataset_fallback_path: str,
        window_size: int,
    ) -> None:
        self.window_size = window_size
        self.buffer: Deque[np.ndarray] = deque(maxlen=window_size)
        self.stats = load_or_compute_stats(stats_path, dataset_fallback_path)

        self.name = Path(model_path).stem.lower()

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_dtype = self.input_details[0]["dtype"]
        self.output_dtype = self.output_details[0]["dtype"]

        self.input_scale, self.input_zero_point = self.input_details[0]["quantization"]
        self.output_scale, self.output_zero_point = self.output_details[0]["quantization"]

        self.temperature, self.smoothing_alpha, self.clip_min, self.clip_max = self._calibration_params(model_path)

        self.prev_probability: Optional[float] = None

    def update(self, sample: Sequence[float]) -> Optional[float]:
        sample_np = np.asarray(sample, dtype=np.float32)

        if sample_np.shape[0] != 4:
            raise ValueError("Each sample must contain 4 features: [HR, RR, SpO2, Movement].")

        self.buffer.append(sample_np)

        if len(self.buffer) < self.window_size:
            return None

        window = np.array(self.buffer, dtype=np.float32)

        # Normalize (HR, RR, SpO2 only — movement untouched)
        window = apply_normalization_single_window(window, self.stats)

        model_input = np.expand_dims(window, axis=0)
        model_input = self._prepare_input(model_input)

        self.interpreter.set_tensor(self.input_details[0]["index"], model_input)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        output = self._dequantize_output(output)

        raw = float(np.asarray(output).reshape(-1)[0])

        prob = self._calibrate_probability(raw)

        return prob

    def _prepare_input(self, model_input: np.ndarray) -> np.ndarray:
        if self.input_dtype in (np.int8, np.uint8):
            if self.input_scale <= 0:
                raise ValueError("Invalid TFLite quantization scale for input tensor.")

            quantized = np.round(model_input / self.input_scale + self.input_zero_point)

            if self.input_dtype == np.int8:
                quantized = np.clip(quantized, -128, 127)
            else:
                quantized = np.clip(quantized, 0, 255)

            return quantized.astype(self.input_dtype)

        return model_input.astype(np.float32)

    def _dequantize_output(self, output: np.ndarray) -> np.ndarray:
        if self.output_dtype in (np.int8, np.uint8):
            if self.output_scale <= 0:
                raise ValueError("Invalid TFLite quantization scale for output tensor.")

            return (output.astype(np.float32) - self.output_zero_point) * self.output_scale

        return output.astype(np.float32)

    @staticmethod
    def _calibration_params(model_path: str) -> tuple[float, float, float, float]:
        model_name = Path(model_path).name.lower()

        if "sepsis" in model_name:
            return 1.45, 0.62, 0.01, 0.95

        if "hypoxia" in model_name:
            return 1.35, 0.62, 0.01, 0.95

        if "arrhythmia" in model_name:
            return 1.20, 0.60, 0.01, 0.95

        return 1.0, 1.0, 0.0, 1.0

    def _calibrate_probability(self, raw_output: float) -> float:
        eps = 1e-6

        # Convert to logit safely
        if 0.0 <= raw_output <= 1.0:
            p = float(np.clip(raw_output, eps, 1.0 - eps))
            logit = float(np.log(p / (1.0 - p)))
        else:
            logit = float(np.clip(raw_output, -30.0, 30.0))

        # Temperature scaling
        scaled = logit / self.temperature

        prob = float(1.0 / (1.0 + np.exp(-np.clip(scaled, -30.0, 30.0))))

        # Clip to range
        prob = float(np.clip(prob, self.clip_min, self.clip_max))

        # EMA smoothing (kept mild)
        if self.prev_probability is not None and self.smoothing_alpha < 1.0:
            prob = self.smoothing_alpha * prob + (1.0 - self.smoothing_alpha) * self.prev_probability
            prob = float(np.clip(prob, self.clip_min, self.clip_max))

        self.prev_probability = prob

        return prob
