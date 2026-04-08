from data_generation.hypoxia_data import WINDOW_SIZE, generate_hypoxia_dataset
from training.common import train_with_distillation


if __name__ == "__main__":
    train_with_distillation(
        disease_name="hypoxia",
        dataset_generator=lambda: generate_hypoxia_dataset(n_normal=2800, n_hypoxia=2800, seed=101),
        window_size=WINDOW_SIZE,
        teacher_epochs=18,
        student_epochs=14,
        batch_size=64,
        random_state=101,
    )
