import pandas as pd
import numpy as np
import glob
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# =====================================
# SETTINGS
# =====================================
WINDOW_SIZE = 200        # Number of keystrokes per training sample
MAX_INTERVAL = 10        # Remove idle pauses greater than 10 seconds

# =====================================
# 1. Load All Session Files
# =====================================
files = glob.glob("session_*.csv")

data_rows = []

for file in files:
    df = pd.read_csv(file, header=None)
    df.columns = ["timestamp", "key", "time_difference"]

    # Convert to numeric safely
    df["time_difference"] = pd.to_numeric(df["time_difference"], errors="coerce")

    intervals = df["time_difference"].dropna()

    # 🚨 Remove extreme idle pauses
    intervals = intervals[intervals < MAX_INTERVAL]

    if len(intervals) < WINDOW_SIZE:
        print(f"Skipping small file: {file}")
        continue

    # =====================================
    # Detect Label From Filename
    # =====================================
    file_lower = file.lower()

    if "frust" in file_lower or "frustrated" in file_lower:
        label = "frustrated"
    elif "calm" in file_lower:
        label = "calm"
    elif "focus" in file_lower or "focused" in file_lower:
        label = "focused"
    else:
        print(f"Label not detected for {file}")
        continue

    # =====================================
    # Window-Based Feature Extraction
    # =====================================
    for start in range(0, len(intervals) - WINDOW_SIZE, WINDOW_SIZE):
        window = intervals.iloc[start:start + WINDOW_SIZE]

        features = {
            "mean_interval": window.mean(),
            "median_interval": window.median(),
            "std_interval": window.std(),
            "pause_1_5_count": (window > 1.5).sum(),
            "pause_5_count": (window > 5).sum(),
            "max_pause": window.max(),
            "total_keystrokes": len(window),
            "label": label
        }

        data_rows.append(features)

# =====================================
# 2. Create Dataset
# =====================================
dataset = pd.DataFrame(data_rows)

print("\nExtracted Feature Dataset (first 5 rows):\n")
print(dataset.head())
print("\nTotal training samples:", len(dataset))

# =====================================
# 3. Prepare Data
# =====================================
X = dataset.drop("label", axis=1)
y = dataset["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =====================================
# 4. Train/Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# =====================================
# 5. Train LightGBM
# =====================================
model = LGBMClassifier(
    n_estimators=100,
    min_data_in_leaf=5,
    num_leaves=15
)

model.fit(X_train, y_train)

# =====================================
# 6. Evaluate
# =====================================
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# =====================================
# 7. Save Model
# =====================================
joblib.dump(model, "keystroke_model.pkl")
print("\nModel saved as keystroke_model.pkl")
