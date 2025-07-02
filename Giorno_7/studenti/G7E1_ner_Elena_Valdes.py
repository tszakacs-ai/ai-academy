from dotenv import load_dotenv
import os
import openai

load_dotenv()

client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY_4o"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT_4o")
)

text = "Mario Rossi ha ricevuto un bonifico sull’IBAN IT60X0542811101000000123456."

prompt = f"""
Trova tutte le entità nel seguente testo (persone, IBAN, ecc.) e restituisci il testo anonimizzato, sostituendo ogni entità con la sua categoria tra parentesi quadre. 
Testo: "{text}"
Risposta solo con il testo anonimizzato.
"""

response = client.chat.completions.create(
    model="gpt-4o", 
    messages=[
        {"role": "system", "content": "Sei un assistente che anonimizza i dati sensibili nei testi."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

anon_text = response.choices[0].message.content
print("Testo anonimizzato:", anon_text)