import openai

# Leggi il file di testo da anonimizzare
with open('./Giorno_6/nota_di_credito.txt', 'r', encoding='utf-8') as f:
    testo = f.read()

# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key="",
    azure_endpoint="",
    api_version="2025-01-01-preview",  # <-- La versione dal portale Azure
)

response = client.chat.completions.create(
    model="gpt-4.1",  # <-- Il nome esatto del deployment in Azure
    messages=[
        {"role": "system", "content": "Sei un analizzatore di documenti professionista. Definisci il tipo del documento che ti fornisco. Rispondi brevemente"},
        {"role": "user", "content": 'Il documento è: '+testo}
    ],
    max_completion_tokens=256,
    temperature=0.2,
)

print(response.choices[0].message.content)