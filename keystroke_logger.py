from pynput import keyboard
import time
import csv
import os

# Create file if not exists
file_name = "session_12.csv"

if not os.path.exists(file_name):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "key", "time_diff"])

last_time = time.time()

def on_press(key):
    global last_time

    current_time = time.time()
    time_diff = current_time - last_time
    last_time = current_time

    try:
        key_value = key.char  # Normal keys
    except AttributeError:
        key_value = str(key)  # Special keys (Key.backspace, Key.enter etc)

    # Store only behavior, not actual content meaning
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_time, key_value, round(time_diff, 4)])

def on_release(key):
    if key == keyboard.Key.esc:
        print("Logging stopped.")
        return False

print("Logging started... Press ESC to stop.")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
