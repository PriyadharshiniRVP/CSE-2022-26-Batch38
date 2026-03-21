import time
import threading
import logging

from nova_assistant import ask_ai

from cute_popup import show_cute_popup


# suppress LightGBM warnings
logging.getLogger("lightgbm").setLevel(logging.CRITICAL)


class FusionEngine:
    def __init__(self, keystroke_logger, facial_logger, stop_event):

        self.W_KEY = 0.65
        self.W_FACE = 0.35
        self.THRESHOLD = 0.6

        self.key_logger = keystroke_logger
        self.face_logger = facial_logger
        self.stop_event = stop_event

        # cooldown timer for AI assistant
        self.last_ai_time = 0
        self.ai_cooldown = 60

    def start(self):

        print("Fusion Engine Running\n")

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

            if frustrated and time.time() - self.last_ai_time > self.ai_cooldown:

                print(" Frustration detected!")

                suggestion = ask_ai(
    "A developer might be stuck while coding. Give a short helpful debugging tip."
)

                # run popup in separate thread so system doesn't block
                threading.Thread(
                    target=show_cute_popup,
                    args=(suggestion,),
                    daemon=True
                ).start()

                self.last_ai_time = time.time()

            time.sleep(1)