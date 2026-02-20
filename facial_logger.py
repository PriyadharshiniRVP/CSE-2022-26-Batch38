import cv2
import time
import json
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting Facial Emotion Monitoring...")
print("Press 'q' or ESC to stop.\n")

# Throttling settings
last_detection_time = 0
detection_interval = 1.0  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Run emotion detection only every 1 second
    if current_time - last_detection_time >= detection_interval:
        last_detection_time = current_time

        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            emotions = result[0]['emotion']
            dominant_emotion = result[0]['dominant_emotion']

            # Define frustration score
            frustration_score = (
                emotions.get('angry', 0) +
                emotions.get('fear', 0) +
                emotions.get('sad', 0)
            )

            output = {
                "timestamp": float(round(current_time, 2)),
                "frustration_score": float(round(frustration_score, 3)),
                "dominant_emotion": str(dominant_emotion)
            }

            # Print JSON safely
            print(json.dumps(output))

        except Exception as e:
            print("Detection error:", e)

    # Show webcam window
    cv2.imshow("Facial Emotion Monitor", frame)

    # Stop control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\nFacial monitoring stopped.")