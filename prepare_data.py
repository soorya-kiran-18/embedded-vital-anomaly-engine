import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------
# LOAD DATA
# -----------------------------
X = np.load("X.npy")   # shape (N, 30, 4)
y = np.load("y.npy")   # shape (N,)

# -----------------------------
# NORMALIZE CONTINUOUS FEATURES
# Features: [HR, RR, SpO2, Movement]
# We normalize only first 3
# -----------------------------
X_norm = X.copy()

for feature_idx in range(3):  # HR, RR, SpO2
    mean = X[:, :, feature_idx].mean()
    std = X[:, :, feature_idx].std()
    X_norm[:, :, feature_idx] = (X[:, :, feature_idx] - mean) / std

# Movement (index 3) is left unchanged

# -----------------------------
# TRAIN / VAL / TEST SPLIT
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_norm, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# -----------------------------
# SAVE PREPARED DATA
# -----------------------------
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Data preparation complete")
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)
