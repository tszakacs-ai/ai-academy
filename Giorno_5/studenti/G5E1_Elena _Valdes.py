import openai
import os
from dotenv import load_dotenv

load_dotenv()

def call_azure_openai_chat(
    system_role: str,
    user_role: str,
    max_tokens: int = 256,
    temperature: float = 1.0
) -> str:
    client = openai.AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY_4o"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT_4o"),
        api_version="2024-12-01-preview",
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_role}
        ],
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    risposta = call_azure_openai_chat(
        system_role="Sei un assistente AI.",
        user_role="Qual è la capitale dell'Italia?",
        max_tokens=256,
        temperature=1.0
    )
    print(risposta)