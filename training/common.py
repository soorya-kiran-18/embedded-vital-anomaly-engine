from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from preprocessing import apply_normalization, compute_normalization_stats, save_stats


class Distiller(tf.keras.Model):
    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        alpha: float = 0.35,
        temperature: float = 4.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            label_smoothing=label_smoothing,
        )
        self.soft_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def compile(self, optimizer: tf.keras.optimizers.Optimizer, metrics: list[tf.keras.metrics.Metric]):
        super().compile(optimizer=optimizer, metrics=metrics)

    def train_step(self, data):
        x_batch, y_batch = data
        y_batch = tf.reshape(tf.cast(y_batch, tf.float32), (-1, 1))
        teacher_logits = self.teacher(x_batch, training=False)
        teacher_soft = tf.nn.sigmoid(teacher_logits / self.temperature)

        with tf.GradientTape() as tape:
            student_logits = self.student(x_batch, training=True)
            student_soft = tf.nn.sigmoid(student_logits / self.temperature)

            student_loss = self.student_loss_fn(y_batch, student_logits)
            distill_loss = self.soft_loss_fn(teacher_soft, student_soft) * (self.temperature ** 2)
            loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        probs = tf.nn.sigmoid(student_logits)
        self.compiled_metrics.update_state(y_batch, probs)
        out = {m.name: m.result() for m in self.metrics}
        out["loss"] = loss
        out["student_loss"] = student_loss
        out["distill_loss"] = distill_loss
        return out

    def test_step(self, data):
        x_batch, y_batch = data
        y_batch = tf.reshape(tf.cast(y_batch, tf.float32), (-1, 1))
        logits = self.student(x_batch, training=False)
        probs = tf.nn.sigmoid(logits)
        loss = self.student_loss_fn(y_batch, logits)
        self.compiled_metrics.update_state(y_batch, probs)
        out = {m.name: m.result() for m in self.metrics}
        out["loss"] = loss
        return out


def _teacher_model(window_size: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(window_size, 4))
    x = layers.Conv1D(48, 5, padding="same", activation="relu")(inputs)
    x = layers.Dropout(0.35)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    logits = layers.Dense(1)(x)
    return models.Model(inputs, logits, name="teacher")


def _student_model(window_size: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(window_size, 4))
    x = layers.Conv1D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.Dropout(0.30)(x)
    x = layers.Conv1D(24, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    logits = layers.Dense(1)(x)
    return models.Model(inputs, logits, name="student")


def _inject_label_noise(y: np.ndarray, noise_rate: float, rng: np.random.Generator) -> np.ndarray:
    y_noisy = y.astype(np.float32, copy=True)
    n_flip = int(noise_rate * len(y_noisy))
    if n_flip <= 0:
        return y_noisy
    idx = rng.choice(len(y_noisy), size=n_flip, replace=False)
    y_noisy[idx] = 1.0 - y_noisy[idx]
    return y_noisy


def _augment_borderline_samples(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    ratio: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    n_new = int(len(x) * ratio)
    if n_new <= 0:
        return x, y

    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    if len(normal_idx) == 0 or len(abnormal_idx) == 0:
        return x, y

    blend_x = []
    blend_y = []
    for _ in range(n_new):
        i = int(rng.choice(normal_idx))
        j = int(rng.choice(abnormal_idx))
        w = float(rng.uniform(0.35, 0.65))
        sample = w * x[i] + (1.0 - w) * x[j]
        sample += rng.normal(0, 0.05, size=sample.shape).astype(np.float32)
        blend_x.append(sample.astype(np.float32))
        blend_y.append(float(rng.integers(0, 2)))  # intentionally ambiguous labels

    x_aug = np.concatenate([x, np.array(blend_x, dtype=np.float32)], axis=0)
    y_aug = np.concatenate([y.astype(np.float32), np.array(blend_y, dtype=np.float32)], axis=0)
    return x_aug, y_aug


def _classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = (probs >= 0.5).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train_with_distillation(
    disease_name: str,
    dataset_generator: Callable[[], tuple[np.ndarray, np.ndarray]],
    window_size: int,
    teacher_epochs: int = 20,
    student_epochs: int = 16,
    batch_size: int = 64,
    random_state: int = 42,
) -> None:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    x_raw, y = dataset_generator()
    x_raw, y = _augment_borderline_samples(x_raw, y, rng=rng, ratio=0.18)

    stats = compute_normalization_stats(x_raw)
    x = apply_normalization(x_raw, stats)
    save_stats(models_dir / f"{disease_name}_norm_stats.npz", stats)
    np.save(models_dir / f"{disease_name}_X.npy", x_raw)
    np.save(models_dir / f"{disease_name}_y.npy", y)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=random_state, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    y_train = _inject_label_noise(y_train, noise_rate=0.10, rng=rng)

    teacher = _teacher_model(window_size)
    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(8e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.04),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_acc", threshold=0.5)],
    )
    teacher.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=teacher_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        ],
    )

    student = _student_model(window_size)
    distiller = Distiller(student=student, teacher=teacher, alpha=0.35, temperature=4.0, label_smoothing=0.05)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(7e-4),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_acc", threshold=0.5)],
    )
    distiller.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=student_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        ],
    )

    # Temperature-scaled inference head for softer probabilities.
    inference_temperature = 1.9
    teacher_infer = tf.keras.Model(
        inputs=teacher.input,
        outputs=tf.keras.layers.Activation("sigmoid")(teacher.output / inference_temperature),
        name=f"{disease_name}_teacher_infer",
    )
    student_infer = tf.keras.Model(
        inputs=student.input,
        outputs=tf.keras.layers.Activation("sigmoid")(student.output / inference_temperature),
        name=f"{disease_name}_student_infer",
    )

    teacher_infer.save(models_dir / f"{disease_name}_teacher.keras")
    student_infer.save(models_dir / f"{disease_name}_student.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(student_infer)
    tflite_model = converter.convert()
    with open(models_dir / f"{disease_name}_student.tflite", "wb") as f:
        f.write(tflite_model)

    test_probs = student_infer.predict(x_test, verbose=0).reshape(-1)
    metrics = _classification_metrics(y_test, test_probs)
    print(f"[{disease_name}] test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"[{disease_name}] Saved:")
    print(f"  {models_dir / f'{disease_name}_teacher.keras'}")
    print(f"  {models_dir / f'{disease_name}_student.keras'}")
    print(f"  {models_dir / f'{disease_name}_student.tflite'}")
    print(f"  {models_dir / f'{disease_name}_norm_stats.npz'}")
