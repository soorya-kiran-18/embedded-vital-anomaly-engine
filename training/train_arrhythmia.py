from data_generation.arrhythmia_data import WINDOW_SIZE, generate_arrhythmia_dataset
from training.common import train_with_distillation


if __name__ == "__main__":
    train_with_distillation(
        disease_name="arrhythmia",
        dataset_generator=lambda: generate_arrhythmia_dataset(n_normal=3200, n_arrhythmia=3200, seed=102),
        window_size=WINDOW_SIZE,
        teacher_epochs=18,
        student_epochs=14,
        batch_size=64,
        random_state=102,
    )
