import time


class FusionEngine:
    def __init__(self, keystroke_logger, facial_logger, stop_event):
        self.W_KEY = 0.65
        self.W_FACE = 0.35
        self.THRESHOLD = 0.6

        self.key_logger = keystroke_logger
        self.face_logger = facial_logger
        self.stop_event = stop_event

    def start(self):
        print("🟢 Fusion Engine Running\n")

        while not self.stop_event.is_set():
            P_key = self.key_logger.get_probability()
            P_face = self.face_logger.get_probability()

            P_total = (self.W_KEY * P_key) + (self.W_FACE * P_face)
            P_total = max(0.0, min(1.0, P_total))

            frustrated = P_total >= self.THRESHOLD

            print({
                "P_keystroke": round(P_key, 3),
                "P_face": round(P_face, 3),
                "P_total": round(P_total, 3),
                "frustrated": frustrated
            })

            time.sleep(1)