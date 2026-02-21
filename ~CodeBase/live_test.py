import time
import numpy as np
import joblib
import json
from pynput import keyboard

# =====================================
# SETTINGS
# =====================================
WINDOW_SIZE = 50
MAX_INTERVAL = 10

# =====================================
# LOAD MODEL
# =====================================
model = joblib.load("keystroke_model.pkl")
le = joblib.load("label_encoder.pkl")

intervals = []
last_time = None

print("🔵 Live Frustration Detection Started...")
print("Press ESC to stop.\n")

# =====================================
# FEATURE EXTRACTION FUNCTION
# =====================================
def extract_features(window):

    window = np.array(window)

    features = [
        window.mean(),
        np.median(window),
        window.std(),
        window.min(),
        window.max(),
        window.max() - window.min(),
        np.sum(window > 1.5),
        np.sum(window > 5),
        np.sum(window < 0.2) / len(window),
        np.sum(window > 2) / len(window),
        np.sum(window < window.mean()) / len(window),
    ]

    return np.array(features).reshape(1, -1)

# =====================================
# KEY PRESS LISTENER
# =====================================
def on_press(key):
    global last_time, intervals

    current_time = time.time()

    if last_time is not None:
        diff = current_time - last_time

        if diff < MAX_INTERVAL:
            intervals.append(diff)

            if len(intervals) >= WINDOW_SIZE:
                window = intervals[-WINDOW_SIZE:]

                features = extract_features(window)

                # ORIGINAL LABEL PREDICTION
                prediction = model.predict(features)
                label = le.inverse_transform(prediction)[0]
                print(f"Prediction: {label}")

                # 🔥 ADDED: Probability for fusion
                probs = model.predict_proba(features)[0]
                frust_index = list(le.classes_).index("frustrated")
                key_prob = float(probs[frust_index])

                with open("keystroke.json", "w") as f:
                    json.dump({"key_prob": key_prob}, f)

    last_time = current_time


def on_release(key):
    if key == keyboard.Key.esc:
        return False


with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()