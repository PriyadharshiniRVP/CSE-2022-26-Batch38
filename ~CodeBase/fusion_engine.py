import json
import time

W_KEY = 0.65
W_FACE = 0.35
THRESHOLD = 0.6

print("🟢 Fusion Engine Running...\n")

while True:
    P_key = 0.0
    P_face = 0.0

    try:
        with open("keystroke.json", "r") as f:
            key_data = json.load(f)
            P_key = float(key_data.get("key_prob", 0.0))
    except:
        pass

    try:
        with open("face.json", "r") as f:
            face_data = json.load(f)
            P_face = float(face_data.get("face_prob", 0.0))
    except:
        pass

    P_total = (W_KEY * P_key) + (W_FACE * P_face)
    P_total = max(0.0, min(1.0, P_total))

    frustrated = P_total >= THRESHOLD

    print({
        "P_keystroke": round(P_key, 3),
        "P_face": round(P_face, 3),
        "P_total": round(P_total, 3),
        "frustrated": frustrated
    })

    time.sleep(1)