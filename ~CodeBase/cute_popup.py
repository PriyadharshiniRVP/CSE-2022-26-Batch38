import tkinter as tk
from ai_assistant import ask_ai


def show_cute_popup(initial_message):

    root = tk.Tk()
    root.title("CodeBuddy 🤖")
    root.geometry("500x400")
    root.configure(bg="#f0f8ff")

    title = tk.Label(
        root,
        text="🤖 CodeBuddy - Debug Assistant",
        font=("Arial", 14, "bold"),
        bg="#f0f8ff"
    )
    title.pack(pady=5)

    # Chat display
    chat_box = tk.Text(root, height=15, width=60, wrap="word")
    chat_box.pack(pady=5)

    chat_box.insert(tk.END, "CodeBuddy:\n" + initial_message + "\n\n")
    chat_box.config(state="disabled")

    # User input
    user_input = tk.Entry(root, width=50)
    user_input.pack(pady=5)

    def send_message():

        question = user_input.get()

        if question.strip() == "":
            return

        chat_box.config(state="normal")
        chat_box.insert(tk.END, "You: " + question + "\n")

        # Ask AI
        response = ask_ai(question)

        chat_box.insert(tk.END, "CodeBuddy: " + response + "\n\n")
        chat_box.config(state="disabled")

        user_input.delete(0, tk.END)

    send_button = tk.Button(
        root,
        text="Ask CodeBuddy",
        command=send_message,
        bg="#87CEFA"
    )
    send_button.pack(pady=5)

    close_button = tk.Button(
        root,
        text="Close",
        command=root.destroy
    )
    close_button.pack(pady=5)

    root.mainloop()