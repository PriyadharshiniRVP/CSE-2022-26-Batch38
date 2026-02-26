import tkinter as tk
from tkinter import ttk
import random


class DevBuddyPopup:

    def __init__(self, emotion_context=None):
        self.emotion_context = emotion_context

        self.root = tk.Tk()
        self.root.title("DevBuddy 💛")
        self.root.attributes("-topmost", True)

        # Soft pastel theme
        self.root.configure(bg="#FFF6F2")

        width = 360
        height = 450

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = screen_width - width - 20
        y = screen_height - height - 60

        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.create_support_tab()
        self.create_review_tab()
        self.create_game_tab()

    # ==========================
    # 💛 SUPPORT TAB
    # ==========================
    def create_support_tab(self):
        support_frame = tk.Frame(self.notebook, bg="#FFF6F2")
        self.notebook.add(support_frame, text="💛 Support")

        self.support_text = tk.Text(
            support_frame,
            height=12,
            wrap="word",
            bg="#FFFFFF",
            font=("Segoe UI", 10),
            bd=0
        )
        self.support_text.pack(padx=10, pady=10, fill="both", expand=True)

        self.support_text.insert(
            "end",
            self.get_soft_intro()
        )
        self.support_text.config(state="disabled")

        breathe_btn = tk.Button(
            support_frame,
            text="Take a 20-sec Breathing Pause 🌿",
            command=self.breathing_message,
            bg="#FFDCDC",
            relief="flat"
        )
        breathe_btn.pack(pady=5)

    def get_soft_intro(self):
        if self.emotion_context == "frustrated":
            return (
                "Hey… I noticed things might feel a little heavy right now.\n\n"
                "Let’s slow down together.\n"
                "You don’t have to solve everything at once.\n\n"
                "Tell me what’s happening. I’m right here 💛"
            )
        else:
            return "Hi there 🌸 How can I support you today?"

    def breathing_message(self):
        self.support_text.config(state="normal")
        self.support_text.insert(
            "end",
            "\n\nLet’s breathe together:\n"
            "Inhale… 1…2…3…4\n"
            "Hold… 1…2…3…4\n"
            "Exhale… 1…2…3…4\n\n"
            "You're doing better than you think 💛"
        )
        self.support_text.config(state="disabled")
        self.support_text.see("end")

    # ==========================
    # 💻 REVIEW TAB
    # ==========================
    def create_review_tab(self):
        review_frame = tk.Frame(self.notebook, bg="#FFF6F2")
        self.notebook.add(review_frame, text="💻 Review")

        self.code_entry = tk.Text(
            review_frame,
            height=10,
            wrap="word",
            bg="#FFFFFF",
            font=("Consolas", 9),
            bd=0
        )
        self.code_entry.pack(padx=10, pady=10, fill="both", expand=True)

        review_btn = tk.Button(
            review_frame,
            text="Gently Review My Code ✨",
            command=self.review_code,
            bg="#DFFFD6",
            relief="flat"
        )
        review_btn.pack(pady=5)

        self.review_output = tk.Label(
            review_frame,
            text="",
            wraplength=300,
            bg="#FFF6F2",
            font=("Segoe UI", 9)
        )
        self.review_output.pack(pady=5)

    def review_code(self):
        code = self.code_entry.get("1.0", "end").strip()

        if not code:
            self.review_output.config(
                text="Maybe paste a small part of your code here, and we’ll look at it together 🌷"
            )
            return

        feedback = self.simple_review_logic(code)
        self.review_output.config(text=feedback)

    def simple_review_logic(self, code):
        if "while True" in code:
            return "I see a 'while True' loop 🌼 Just make sure there’s a safe exit condition."

        if "==" in code and "if" not in code:
            return "There’s a comparison here. Maybe check if it’s inside a proper condition block?"

        if "print(" in code:
            return "Print statements are helpful 🌷 Later you might consider logging for larger systems."

        return "It looks structurally okay at a glance 🌸 Want to walk me through what it's supposed to do?"

    # ==========================
    # 🎮 BUG SMASHER GAME
    # ==========================
    def create_game_tab(self):
        game_frame = tk.Frame(self.notebook, bg="#FFF6F2")
        self.notebook.add(game_frame, text="🎮 Bug Smasher")

        self.score = 0

        self.score_label = tk.Label(
            game_frame,
            text="Score: 0",
            font=("Segoe UI", 11),
            bg="#FFF6F2"
        )
        self.score_label.pack(pady=10)

        self.bug_button = tk.Button(
            game_frame,
            text="🐞",
            font=("Arial", 30),
            command=self.smash_bug,
            relief="flat"
        )
        self.bug_button.pack(expand=True)

        tip_label = tk.Label(
            game_frame,
            text="Smash a few bugs and reset your brain 🌿",
            bg="#FFF6F2",
            font=("Segoe UI", 9)
        )
        tip_label.pack(pady=5)

    def smash_bug(self):
        self.score += 1
        self.score_label.config(text=f"Score: {self.score}")

        x = random.randint(20, 250)
        y = random.randint(50, 200)

        self.bug_button.place(x=x, y=y)

    # ==========================
    def show(self):
        self.root.mainloop()