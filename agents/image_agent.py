from utils.llm import generate_response


import requests
import os

def clean_ascii(text: str) -> str:
    return text.encode("ascii", "ignore").decode()

HF_API_KEY = clean_ascii(os.getenv("HF_API_KEY", ""))

def clean_text(text: str) -> str:
    """
    Remove problematic unicode characters
    """
    return text.encode("ascii", "ignore").decode()

def generate_image(prompt):
    # API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-2"
    # API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
    API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    def clean_ascii(text: str) -> str:
        return text.encode("ascii", "ignore").decode()

    clean_prompt = clean_ascii(prompt)

    headers = {
        "Authorization": f"Bearer {clean_ascii(HF_API_KEY)}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": clean_prompt,
        "options": {"wait_for_model": True}
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        with open("output.png", "wb") as f:
            f.write(response.content)
        return "output.png"

    return f"Error: {response.text}"


def run(query: str, context: str) -> str:
    """
    Image Agent with prompt engineering
    """

    # Step 1: Convert user query into optimized image prompt
    prompt_engineering = f"""
You are an expert prompt engineer for AI image generation.

User request:
{query}

Create a detailed image generation prompt:
- Include style (realistic, illustration, etc.)
- Include lighting, composition
- Include context and details
- Make it vivid and specific

Output only the prompt.
"""

    image_prompt = generate_response(prompt_engineering)
    image_prompt = image_prompt.replace("“", '"').replace("”", '"')
    image_path = generate_image(image_prompt)

    return {
        "type": "image",
        "prompt": image_prompt,
        "path": image_path
    }

