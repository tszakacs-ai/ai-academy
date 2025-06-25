import openai

# Leggi il file di testo da anonimizzare
with open('./Giorno_5/nota_di_credito.txt', 'r', encoding='utf-8') as f:
    testo = f.read()

# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key="",  # <-- La tua chiave API di Azure OpenAI
    azure_endpoint="", # <-- Il tuo  API endpoint di Azure OpenAI
    api_version="2025-01-01-preview",  # <-- La versione dal portale Azure
)

response = client.chat.completions.create(
    model="gpt-4.1",  # <-- Il nome esatto del deployment in Azure
    messages=[
        {"role": "system", "content": "Sei anominizzatore professionista. Devi anonimizzare il testo "
        "che ti verrà fornito.Restituisci il testo con al posto dei valori, i valori anonimi come [NOME], "
        "[IBAN], [CODICE_FISCALE] e [TELEFONO]."},
        {"role": "user", "content": testo}
    ],
    max_completion_tokens=256,
    temperature=1,
)

print(response.choices[0].message.content)