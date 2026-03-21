import warnings
warnings.filterwarnings("ignore")

import threading
import time

from facial_logger import FacialLogger
from keystroke_logger import KeystrokeLogger
from fusion_engine import FusionEngine


def main():

    print(" Starting Frustration Detection System...")

    stop_event = threading.Event()

    # Initialize modules
    facial_logger = FacialLogger(stop_event)
    keystroke_logger = KeystrokeLogger(stop_event)

    fusion_engine = FusionEngine(
        keystroke_logger,
        facial_logger,
        stop_event
    )

    # Threads
    face_thread = threading.Thread(target=facial_logger.start)
    key_thread = threading.Thread(target=keystroke_logger.start)
    fusion_thread = threading.Thread(target=fusion_engine.start)

    face_thread.start()
    key_thread.start()
    fusion_thread.start()

    print(" Facial Emotion Monitoring Started")
    print(" Keystroke Monitoring Started")
    print(" Fusion Engine Running")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:

        print("\n Stopping entire system...")

        stop_event.set()

        face_thread.join()
        key_thread.join()
        fusion_thread.join()

        print(" System stopped cleanly.")


if __name__ == "__main__":
    main()