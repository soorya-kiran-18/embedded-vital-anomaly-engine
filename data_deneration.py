import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
WINDOW_SIZE = 30      # 30 seconds
NUM_FEATURES = 4      # HR, RR, SpO2, Movement
NUM_NORMAL = 500
NUM_APNEA = 500

np.random.seed(42)

# -----------------------------
# NORMAL SAMPLE GENERATION
# -----------------------------
def generate_normal_sample():
    hr_base = np.random.uniform(65, 85)
    rr_base = np.random.uniform(14, 18)
    spo2_base = np.random.uniform(97, 99)

    hr = hr_base + np.random.normal(0, 1.5, WINDOW_SIZE)
    rr = rr_base + np.random.normal(0, 0.8, WINDOW_SIZE)
    spo2 = spo2_base + np.random.normal(0, 0.3, WINDOW_SIZE)

    # Movement: mostly still, occasional movement
    movement = np.random.choice([0, 1], size=WINDOW_SIZE, p=[0.9, 0.1])

    sample = np.column_stack((hr, rr, spo2, movement))
    return sample


# -----------------------------
# APNEA-LIKE SAMPLE GENERATION
# -----------------------------
def generate_apnea_sample():
    hr_base = np.random.uniform(65, 80)
    rr_base = np.random.uniform(14, 18)
    spo2_base = np.random.uniform(96, 98)

    hr = []
    rr = []
    spo2 = []

    apnea_start = np.random.randint(8, 15)

    current_hr = hr_base
    current_rr = rr_base
    current_spo2 = spo2_base

    for t in range(WINDOW_SIZE):
        if t >= apnea_start:
            # RR drops sharply
            current_rr = max(0, current_rr - np.random.uniform(0.8, 1.5))
            # SpO2 drops slowly
            current_spo2 -= np.random.uniform(0.1, 0.3)
            # HR mild fluctuation
            current_hr += np.random.normal(0.3, 0.6)
        else:
            current_hr += np.random.normal(0, 1.0)
            current_rr += np.random.normal(0, 0.6)
            current_spo2 += np.random.normal(0, 0.2)

        hr.append(current_hr)
        rr.append(current_rr)
        spo2.append(current_spo2)

    # Movement: almost no movement
    movement = np.zeros(WINDOW_SIZE)

    sample = np.column_stack((hr, rr, spo2, movement))
    return sample


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

# Shuffle dataset
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Save to disk
np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset generated:")
print("X shape:", X.shape)
print("y shape:", y.shape)
