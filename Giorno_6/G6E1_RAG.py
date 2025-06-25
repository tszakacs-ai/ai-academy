import openai
import os

client = openai.AzureOpenAI(
    api_key="LA_TUA_API_KEY",  
    azure_endpoint="IL_TUO_API_ENDPOINT", 
    api_version="2024-12-01-preview",  
)

DOCUMENT_TYPE_PROMPT = """
Ti fornirò un documento aziendale.

Il tuo compito è determinare il tipo di documento, scegliendo tra i seguenti: 
- Mail
- Nota di credito
- Ordine di acquisto
- Contratto

Devi basarti solo sul contenuto del documento fornito.

Ora incollerò il contenuto del documento tra tripli apici. Rispondi semplicemente con il tipo, nulla di più.

Documento:
'''
{documento}
'''
"""

def ask_model(prompt):
    response = client.chat.completions.create(
    model="o4-mini",  
    messages=[
        {"role": "system", "content": "Sei un assistente AI."},
        {"role": "user", "content": prompt}
    ],
    max_completion_tokens=256,
    temperature=1,
    )
    return response.choices[0].message.content

def test_documents():
    for file in os.listdir("documents"):
        with open(os.path.join("documents", file), "r", encoding="utf-8") as f:
            text = f.read()
        prompt = DOCUMENT_TYPE_PROMPT.format(documento=text)
        response = ask_model(prompt)
        print(f"[{file}] → Tipo rilevato: {response}")

if __name__ == "__main__":
    test_documents()
