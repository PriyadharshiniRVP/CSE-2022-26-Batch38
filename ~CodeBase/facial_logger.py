import cv2
import time
import json
import numpy as np
from collections import deque
from deepface import DeepFace

cap = cv2.VideoCapture(0)

print("🟢 Starting Facial Emotion Monitoring...")
print("Press 'q' or ESC to stop.\n")

last_detection_time = 0
detection_interval = 1.0
smoothing_window = 7

buffer = deque(maxlen=smoothing_window)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    if current_time - last_detection_time >= detection_interval:
        last_detection_time = current_time

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

            buffer.append(float(raw_score))
            smoothed_score = np.mean(buffer)

            face_prob = min(smoothed_score / 100.0, 1.0)

            print({"face_prob": round(face_prob, 3)})

            # 🔥 ADDED: Write to file
            with open("face.json", "w") as f:
                json.dump({"face_prob": float(face_prob)}, f)

        except Exception as e:
            print("Detection error:", e)

    cv2.imshow("Facial Emotion Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\n🔴 Facial monitoring stopped.")