import os
from openai import OpenAI

# Groq uses OpenAI-compatible API
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def generate_response(prompt: str) -> str:
    """
    Generic LLM call using Groq
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Groq model
            messages=[
                {"role": "system", "content": "You are a helpful business AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"