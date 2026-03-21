import time
import numpy as np
import joblib
import logging
from pynput import keyboard

# Suppress LightGBM warnings properly
logging.getLogger("lightgbm").setLevel(logging.ERROR)


class KeystrokeLogger:
    def __init__(self, stop_event):
        self.stop_event = stop_event

        self.WINDOW_SIZE = 50
        self.MAX_INTERVAL = 10

        self.model = joblib.load("../Models/keystroke_model.pkl")
        self.le = joblib.load("../Models/label_encoder.pkl")

        self.intervals = []
        self.last_time = None
        self.key_prob = 0.0

    def extract_features(self, window):
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

    def on_press(self, key):
        current_time = time.time()

        if self.last_time is not None:
            diff = current_time - self.last_time

            if diff < self.MAX_INTERVAL:
                self.intervals.append(diff)

                if len(self.intervals) >= self.WINDOW_SIZE:
                    window = self.intervals[-self.WINDOW_SIZE:]
                    features = self.extract_features(window)

                    probs = self.model.predict_proba(features)[0]
                    frust_index = list(self.le.classes_).index("frustrated")
                    self.key_prob = float(probs[frust_index])

        self.last_time = current_time

    def start(self):
        print("Keystroke Monitoring Started")

        with keyboard.Listener(on_press=self.on_press) as listener:
            while not self.stop_event.is_set():
                time.sleep(0.1)
            listener.stop()

    def get_probability(self):
        return self.key_prob

        