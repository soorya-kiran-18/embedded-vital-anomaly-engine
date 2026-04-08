from data_generation.sepsis_data import WINDOW_SIZE, generate_sepsis_dataset
from training.common import train_with_distillation


if __name__ == "__main__":
    train_with_distillation(
        disease_name="sepsis",
        dataset_generator=lambda: generate_sepsis_dataset(n_normal=2600, n_sepsis=2600, seed=103),
        window_size=WINDOW_SIZE,
        teacher_epochs=20,
        student_epochs=16,
        batch_size=64,
        random_state=103,
    )
