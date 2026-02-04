# Install: pip install pynput
from pynput import keyboard, mouse
import time
import json

log = []

def on_press(key):
    log.append({'type': 'key_press', 'key': str(key), 'time': time.time()})

def on_release(key):
    # Stop listeners and exit on pressing the ESC key
    if key == keyboard.Key.esc:
        print("ESC pressed. Stopping recording...")
        # Stop the mouse listener
        mouse_listener.stop()
        # Stop the keyboard listener by returning False
        return False
    log.append({'type': 'key_release', 'key': str(key), 'time': time.time()})

def on_click(x, y, button, pressed):
    log.append({'type': 'mouse_click', 'x': x, 'y': y, 'button': str(button), 'pressed': pressed, 'time': time.time()})

# Start listeners
print("Recording... Press ESC key to stop and save.")
mouse_listener = mouse.Listener(on_click=on_click)
key_listener = keyboard.Listener(on_press=on_press, on_release=on_release)

mouse_listener.start()
key_listener.start()

# This will block the program until the keyboard listener stops
# (which happens when on_release returns False for the ESC key)
key_listener.join()

# Stop the mouse listener explicitly (it should already be stopped, but this is safe)
mouse_listener.stop()

# Save log to file
with open('session_log.json', 'w') as f:
    json.dump(log, f)
print(f"Recording saved. Logged {len(log)} events to 'session_log.json'.")