import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
WINDOW_SIZE = 30
NUM_FEATURES = 4
NUM_NORMAL = 1200
NUM_APNEA = 1200

np.random.seed(42)

# -----------------------------
# NORMAL SAMPLE GENERATION
# -----------------------------
def generate_normal_sample():

    hr_base = np.random.uniform(60, 90)
    rr_base = np.random.uniform(11, 20)
    spo2_base = np.random.uniform(94, 99)

    hr = hr_base + np.random.normal(0, 2.0, WINDOW_SIZE)
    rr = rr_base + np.random.normal(0, 1.5, WINDOW_SIZE)
    spo2 = spo2_base + np.random.normal(0, 0.8, WINDOW_SIZE)

    # 20% chance of structured mild dip (still normal)
    if np.random.rand() < 0.2:
        event_start = np.random.randint(6, 20)
        duration = np.random.randint(2, 5)

        for t in range(event_start, min(WINDOW_SIZE, event_start + duration)):
            rr[t] -= np.random.uniform(1.0, 2.0)
            spo2[t] -= np.random.uniform(0.5, 1.0)
            hr[t] += np.random.normal(0.5, 0.5)

    movement = np.random.choice([0, 1], size=WINDOW_SIZE, p=[0.85, 0.15])

    return np.column_stack((hr, rr, spo2, movement))


# -----------------------------
# APNEA SAMPLE GENERATION
# -----------------------------
def generate_apnea_sample():

    hr_base = np.random.uniform(60, 85)
    rr_base = np.random.uniform(11, 18)
    spo2_base = np.random.uniform(93, 98)

    hr = hr_base + np.random.normal(0, 2.0, WINDOW_SIZE)
    rr = rr_base + np.random.normal(0, 1.5, WINDOW_SIZE)
    spo2 = spo2_base + np.random.normal(0, 0.8, WINDOW_SIZE)

    apnea_start = np.random.randint(6, 18)
    apnea_duration = np.random.randint(3, 8)

    for t in range(apnea_start, min(WINDOW_SIZE, apnea_start + apnea_duration)):

        severity = np.random.choice(
            ["mild", "moderate", "severe"],
            p=[0.4, 0.4, 0.2]
        )

        if severity == "mild":
            rr[t] -= np.random.uniform(1.5, 3.0)
            spo2[t] -= np.random.uniform(0.5, 1.2)

        elif severity == "moderate":
            rr[t] -= np.random.uniform(3.0, 5.0)
            spo2[t] -= np.random.uniform(1.0, 2.0)

        else:  # severe
            rr[t] -= np.random.uniform(5.0, 7.0)
            spo2[t] -= np.random.uniform(2.0, 3.0)

        hr[t] += np.random.normal(1.0, 1.5)

    movement = np.random.choice([0, 1], size=WINDOW_SIZE, p=[0.85, 0.15])

    return np.column_stack((hr, rr, spo2, movement))


# -----------------------------
# DATASET CREATION
# -----------------------------
X = []
y = []

for _ in range(NUM_NORMAL):
    X.append(generate_normal_sample())
    y.append(0)

for _ in range(NUM_APNEA):
    X.append(generate_apnea_sample())
    y.append(1)

X = np.array(X)
y = np.array(y)

indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset generated successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)