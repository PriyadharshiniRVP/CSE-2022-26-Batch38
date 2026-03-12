import cv2
import time
import numpy as np
from collections import deque
from deepface import DeepFace


class FacialLogger:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.cap = cv2.VideoCapture(0)
        self.last_detection_time = 0
        self.detection_interval = 1.0
        self.buffer = deque(maxlen=7)
        self.face_prob = 0.0

    def start(self):
        print("🟢 Facial Emotion Monitoring Started")

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = time.time()

            if current_time - self.last_detection_time >= self.detection_interval:
                self.last_detection_time = current_time

                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False
                    )

                    emotions = result[0]['emotion']

                    raw_score = (
                        0.5 * emotions.get('angry', 0) +
                        0.3 * emotions.get('sad', 0) +
                        0.2 * emotions.get('fear', 0)
                    )

                    self.buffer.append(float(raw_score))
                    smoothed = np.mean(self.buffer)
                    self.face_prob = min(smoothed / 100.0, 1.0)

                except:
                    pass

            cv2.imshow("Facial Emotion Monitor", frame)

            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                self.stop_event.set()

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_probability(self):
        return self.face_prob