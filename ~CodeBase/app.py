import warnings
warnings.filterwarnings("ignore")

import threading
import time
import subprocess
import requests

from facial_logger import FacialLogger
from keystroke_logger import KeystrokeLogger
from fusion_engine import FusionEngine


def start_ollama():

    try:
        requests.get("http://localhost:11434")
        print("🟢 Ollama already running")
    except:
        print("🚀 Starting Ollama...")
        subprocess.Popen(["ollama", "serve"])
        time.sleep(5)


def main():

    print("🚀 Starting system...")

    start_ollama()

    stop_event = threading.Event()

    facial_logger = FacialLogger(stop_event)
    keystroke_logger = KeystrokeLogger(stop_event)

    fusion_engine = FusionEngine(
        keystroke_logger,
        facial_logger,
        stop_event
    )

    face_thread = threading.Thread(target=facial_logger.start)
    key_thread = threading.Thread(target=keystroke_logger.start)
    fusion_thread = threading.Thread(target=fusion_engine.start)

    face_thread.start()
    key_thread.start()
    fusion_thread.start()

    print("🟢 Facial Emotion Monitoring Started")
    print("🔵 Keystroke Monitoring Started")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:

        print("\n🛑 Stopping entire system...")

        stop_event.set()

        face_thread.join()
        key_thread.join()
        fusion_thread.join()

        print("✅ System stopped cleanly.")


if __name__ == "__main__":
    main()