import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def ask_ai(prompt):

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "codellama",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        return response.json()["response"]

    except Exception as e:
        print("⚠️ AI Assistant unavailable:", e)
        return "Take a deep breath. Try isolating the bug step-by-step. You've got this."


def debugging_assistant(problem):

    prompt = f"""
You are a friendly AI coding assistant helping a frustrated developer.

Problem:
{problem}

Give:
1 short explanation
1 debugging suggestion
1 encouragement message.

Keep the answer under 4 lines.
"""

    return ask_ai(prompt)