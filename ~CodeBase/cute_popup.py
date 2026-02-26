import tkinter as tk
from tkinter import messagebox

def show_cute_popup():

    root = tk.Tk()
    root.title("DevBuddy 💛")

    root.geometry("350x300")
    root.configure(bg="#FFF6F2")

    tk.Label(
        root,
        text="Hey… you seem a little stressed 🌸",
        bg="#FFF6F2",
        fg="#444",
        font=("Segoe UI", 12)
    ).pack(pady=10)

    tk.Label(
        root,
        text="Let's slow down together 💛",
        bg="#FFF6F2",
        fg="#FF7A8A",
        font=("Segoe UI", 11)
    ).pack()

    def breathing():
        messagebox.showinfo(
            "Breathing Time 🌿",
            "Inhale... 1 2 3 4\nHold... 1 2 3 4\nExhale... 1 2 3 4\n\nYou're doing great 💛"
        )

    tk.Button(
        root,
        text="20-sec Breathing Pause 🌿",
        bg="#FFDCDC",
        command=breathing
    ).pack(pady=10)

    score = 0

    def bug_smasher():
        nonlocal score
        score += 1
        bug_label.config(text=f"🐞 Bugs Smashed: {score}")

    tk.Button(
        root,
        text="Smash a Bug 🐞",
        bg="#FFC4C4",
        command=bug_smasher
    ).pack(pady=5)

    bug_label = tk.Label(
        root,
        text="🐞 Bugs Smashed: 0",
        bg="#FFF6F2"
    )
    bug_label.pack()

    tk.Button(
        root,
        text="I'm Okay Now 💛",
        bg="#FFB6B9",
        command=root.destroy
    ).pack(pady=15)

    root.mainloop()