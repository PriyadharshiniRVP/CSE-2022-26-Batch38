import threading
import time
from cute_popup import show_cute_popup
from pynput import keyboard
from facial_logger import FacialLogger
from keystroke_logger import KeystrokeLogger
from fusion_engine import FusionEngine


def main():
    stop_event = threading.Event()

    facial = FacialLogger(stop_event)
    keystroke = KeystrokeLogger(stop_event)
    fusion = FusionEngine(keystroke, facial, stop_event)

    t1 = threading.Thread(target=facial.start)
    t2 = threading.Thread(target=keystroke.start)
    t3 = threading.Thread(target=fusion.start)

    t1.start()
    t2.start()
    t3.start()

    print("\nPress ESC to stop entire monitoring...\n")

    def on_press(key):
        if key == keyboard.Key.esc:
            print("\n🛑 Stopping entire system...")
            stop_event.set()
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    t1.join()
    t2.join()
    t3.join()

    print("✅ System stopped cleanly.")


if __name__ == "__main__":
    main()