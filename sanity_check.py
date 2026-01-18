import numpy as np
import matplotlib.pyplot as plt

# Load generated data
X = np.load("X.npy")
y = np.load("y.npy")

# Pick one normal and one apnea sample
normal_idx = np.where(y == 0)[0][0]
apnea_idx = np.where(y == 1)[0][0]

normal_sample = X[normal_idx]
apnea_sample = X[apnea_idx]

time = np.arange(30)

def plot_sample(sample, title):
    plt.figure(figsize=(10, 6))

    plt.subplot(4, 1, 1)
    plt.plot(time, sample[:, 0])
    plt.ylabel("HR")

    plt.subplot(4, 1, 2)
    plt.plot(time, sample[:, 1])
    plt.ylabel("RR")

    plt.subplot(4, 1, 3)
    plt.plot(time, sample[:, 2])
    plt.ylabel("SpO₂")

    plt.subplot(4, 1, 4)
    plt.step(time, sample[:, 3])
    plt.ylabel("Move")
    plt.xlabel("Time (sec)")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Plot both
plot_sample(normal_sample, "NORMAL SAMPLE")
plot_sample(apnea_sample, "APNEA-LIKE SAMPLE")
