from dotenv import load_dotenv
import os
import openai

load_dotenv()

client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY_4o"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT_4o")
)

text = "Leonardo da Vinci ha dipinto la Gioconda a Firenze."

prompt = f"""
Trova tutte le entità nel seguente testo (persone, opere, città, ecc.) e restituisci una lista nel formato: entità → tipo.
Testo: "{text}"
Rispondi solo con la lista.
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Sei un assistente che estrae entità dai testi."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

print(response.choices[0].message.content)