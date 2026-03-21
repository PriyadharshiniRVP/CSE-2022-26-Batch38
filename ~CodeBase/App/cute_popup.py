import customtkinter as ctk
from nova_assistant import ask_ai
import threading
import time

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def show_cute_popup(initial_message):

    root = ctk.CTk()
    root.title("CodeBuddy 🤖")
    root.geometry("720x600")

    root.configure(fg_color="#0f172a")

    # ================= HEADER =================

    header = ctk.CTkFrame(root, fg_color="#111827", corner_radius=0)
    header.pack(fill="x")

    title = ctk.CTkLabel(
        header,
        text="🤖 CodeBuddy",
        font=("Segoe UI", 26, "bold")
    )
    title.pack(pady=(15, 0))

    subtitle = ctk.CTkLabel(
        header,
        text="Your AI Debug Companion",
        font=("Segoe UI", 13),
        text_color="#9ca3af"
    )
    subtitle.pack(pady=(0, 15))

    # ================= CHAT AREA =================

    chat_frame = ctk.CTkScrollableFrame(
        root,
        fg_color="#020617",
        corner_radius=12
    )

    chat_frame.pack(fill="both", expand=True, padx=20, pady=15)

    # ================= ADD MESSAGE FUNCTION =================

    def add_message(text, sender):

        bubble_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        bubble_frame.pack(fill="x", pady=10)

        if sender == "user":

            bubble = ctk.CTkLabel(
                bubble_frame,
                text=text,
                wraplength=480,
                justify="right",
                fg_color="#2563eb",
                text_color="white",
                corner_radius=18,
                padx=14,
                pady=10
            )

            bubble.pack(anchor="e", padx=10)

        else:

            bubble = ctk.CTkLabel(
                bubble_frame,
                text=text,
                wraplength=480,
                justify="left",
                fg_color="#1e293b",
                text_color="white",
                corner_radius=18,
                padx=14,
                pady=10
            )

            bubble.pack(anchor="w", padx=10)

        root.update_idletasks()

        # auto scroll to bottom
        chat_frame._parent_canvas.yview_moveto(1)

    # ================= TYPING INDICATOR =================

    def show_typing():

        bubble_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        bubble_frame.pack(fill="x", pady=10)

        typing_label = ctk.CTkLabel(
            bubble_frame,
            text="CodeBuddy is thinking",
            fg_color="#1e293b",
            corner_radius=18,
            padx=14,
            pady=10
        )

        typing_label.pack(anchor="w", padx=10)

        for i in range(3):
            for dots in [" .", " ..", " ..."]:
                typing_label.configure(text="CodeBuddy is thinking" + dots)
                root.update()
                time.sleep(0.4)

        bubble_frame.destroy()

    # ================= INITIAL MESSAGE =================

    add_message(initial_message, "bot")

    # ================= INPUT AREA =================

    input_frame = ctk.CTkFrame(
        root,
        fg_color="#111827",
        corner_radius=20
    )

    input_frame.pack(fill="x", padx=20, pady=(0, 15))

    user_input = ctk.CTkEntry(
        input_frame,
        placeholder_text="Ask CodeBuddy something...",
        height=42
    )

    user_input.pack(side="left", fill="x", expand=True, padx=10, pady=10)

    # ================= SEND MESSAGE =================

    def process_ai(question):

        show_typing()

        response = ask_ai(question)

        add_message(response, "bot")

    def send_message():

        question = user_input.get().strip()

        if question == "":
            return

        add_message(question, "user")

        user_input.delete(0, "end")

        threading.Thread(target=process_ai, args=(question,)).start()

    user_input.bind("<Return>", lambda e: send_message())

    send_button = ctk.CTkButton(
        input_frame,
        text="➤",
        font=("Segoe UI", 18, "bold"),
        width=50,
        corner_radius=20,
        command=send_message
    )

    send_button.pack(side="right", padx=10)

    root.mainloop()