import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import joblib

from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =====================================
# SETTINGS
# =====================================
WINDOW_SIZE = 50
MAX_INTERVAL = 10

# =====================================
# 1. LOAD ALL SESSION FILES
# =====================================
files = glob.glob("session_*.csv")

data_rows = []

for file in files:
    df = pd.read_csv(file, header=None)
    df.columns = ["timestamp", "key", "time_difference"]

    df["time_difference"] = pd.to_numeric(df["time_difference"], errors="coerce")

    intervals = df["time_difference"].dropna()
    intervals = intervals[intervals < MAX_INTERVAL]

    if len(intervals) < WINDOW_SIZE:
        print(f"Skipping small file: {file}")
        continue

    file_lower = file.lower()

    if "frust" in file_lower:
        label = "frustrated"
    elif "normal" in file_lower:
        label = "normal"
    else:
        print(f"Skipping unlabeled file: {file}")
        continue

    # =====================================
    # WINDOW FEATURE EXTRACTION
    # =====================================
    for start in range(0, len(intervals) - WINDOW_SIZE, WINDOW_SIZE):
        window = intervals.iloc[start:start + WINDOW_SIZE]

        features = {
            "mean_interval": window.mean(),
            "median_interval": window.median(),
            "std_interval": window.std(),
            "min_interval": window.min(),
            "max_interval": window.max(),
            "range_interval": window.max() - window.min(),
            "pause_1_5_count": (window > 1.5).sum(),
            "pause_5_count": (window > 5).sum(),
            "fast_typing_ratio": (window < 0.2).sum() / len(window),
            "slow_typing_ratio": (window > 2).sum() / len(window),
            "burst_ratio": (window < window.mean()).sum() / len(window),
            "label": label
        }

        data_rows.append(features)

# =====================================
# 2. CREATE DATASET
# =====================================
dataset = pd.DataFrame(data_rows)

if len(dataset) == 0:
    print("❌ No training samples found.")
    exit()

print("\nTotal training samples:", len(dataset))
print("\nLabel distribution:")
print(dataset["label"].value_counts())

if dataset["label"].nunique() < 2:
    print("❌ Need both normal and frustrated sessions.")
    exit()

# =====================================
# 3. PREPARE DATA
# =====================================
X = dataset.drop("label", axis=1)
y = dataset["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =====================================
# 4. TRAIN / TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

# =====================================
# 5. TRAIN LIGHTGBM (IMPROVED)
# =====================================
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_data_in_leaf=10,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =====================================
# 6. CROSS VALIDATION
# =====================================
cv_scores = cross_val_score(model, X, y_encoded, cv=5)
print("\nCross Validation Accuracy:", cv_scores.mean())

# =====================================
# 7. EVALUATION
# =====================================
y_pred = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =====================================
# 8. FEATURE IMPORTANCE
# =====================================
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 6))
plt.barh(feature_names, importances)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# =====================================
# 9. SAVE MODEL
# =====================================
joblib.dump(model, "keystroke_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n✅ Model saved successfully.")
