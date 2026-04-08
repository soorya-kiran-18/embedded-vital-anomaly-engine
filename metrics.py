import math

from engine.detection_engine import DetectionEngine, simulate_realtime_sample


CONDITIONS = ("apnea", "hypoxia", "arrhythmia", "sepsis")
EVENT_WINDOWS = {
    "apnea": (90, 119),
    "hypoxia": (160, 204),
    "arrhythmia": (230, 249),
    "sepsis": (300, 419),
}


def ground_truth_label(timestep):
    for label, (start, end) in EVENT_WINDOWS.items():
        if start <= timestep <= end:
            return label
    return "none"


def binary_cross_entropy(y_true, probability, eps=1e-7):
    p = min(max(float(probability), eps), 1.0 - eps)
    return -(y_true * math.log(p) + (1 - y_true) * math.log(1.0 - p))


def evaluate_engine(num_steps=400):
    engine = DetectionEngine()
    total_loss = 0.0
    tp = 0
    fp = 0
    fn = 0

    try:
        for t in range(num_steps):
            sample = simulate_realtime_sample(t)
            output = engine.process_sample(sample)
            truth = ground_truth_label(t)
            predicted = output["dominant"]["state"].lower()

            for name in CONDITIONS:
                y_true = 1 if truth == name else 0
                probability = output[name]["probability"]
                if probability is None:
                    probability = 0.0
                total_loss += binary_cross_entropy(y_true, probability)

            if predicted == truth and truth != "none":
                tp += 1
            elif truth == "none" and predicted != "none":
                fp += 1
            elif truth != "none" and predicted == "none":
                fn += 1
            elif truth != "none" and predicted != truth:
                fp += 1
                fn += 1

    finally:
        engine.shutdown()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    average_loss = total_loss / (num_steps * len(CONDITIONS))

    print("Evaluation Results")
    print("==================")
    print(f"Timesteps:       {num_steps}")
    print(f"Total loss:      {total_loss:.6f}")
    print(f"Average loss:    {average_loss:.6f}")
    print(f"True positives:  {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")


if __name__ == "__main__":
    evaluate_engine()
