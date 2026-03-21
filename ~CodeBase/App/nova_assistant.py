import boto3
import json

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)


SYSTEM_PROMPT = """
You are CodeBuddy, an experienced developer helping another programmer.

Your personality:
- Friendly and practical
- Think like a real developer reading someone else's code
- Speak naturally, like a teammate helping debug
- Avoid robotic phrases like "Certainly", "As an AI", or "Here is the explanation"

How to respond:
• If the user sends code, analyze it and point out possible issues
• Explain WHY the issue happens
• Suggest a fix or debugging approach
• If useful, show a corrected code snippet

Tone:
- Conversational
- Clear
- Helpful
- Encouraging

Avoid long academic explanations. Focus on solving the problem.
"""


def ask_ai(user_input):

    try:

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"{SYSTEM_PROMPT}\n\nUser message:\n{user_input}"
                        }
                    ]
                }
            ]
        }

        response = bedrock.invoke_model(
            modelId="global.amazon.nova-2-lite-v1:0",
            body=json.dumps(body),
            contentType="application/json"
        )

        result = json.loads(response["body"].read())

        answer = result["output"]["message"]["content"][0]["text"]

        return answer.strip()

    except Exception as e:

        print("Nova AI unavailable:", e)

        return (
            "Hmm, something went wrong with the AI call.\n\n"
            "Quick debugging idea:\n"
            "Try printing intermediate values or isolating the part of the code where the bug appears."
        )