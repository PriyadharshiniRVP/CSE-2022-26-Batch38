import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def ask_ai(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


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