import openai

# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key="1hXcKdMliWRMC5rzKTY3omTf1PDcSwqfE7IcJnhAlqcJHX2E4NUIJQQJ99BFACfhMk5XJ3w3AAAAACOGg77H",  # <-- La tua chiave API di Azure OpenAI
    azure_endpoint="https://andre-mc4pjn8f-swedencentral.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview", # <-- Il tuo  API endpoint di Azure OpenAI
    api_version="2024-12-01-preview",  # <-- La versione dal portale Azure
)

response = client.chat.completions.create(
    model="o4-mini",  # <-- Il nome esatto del deployment in Azure
    messages=[
        {"role": "system", "content": "Sei un assistente AI."},
        {"role": "user", "content": "Qual è la capitale dell'Italia?"}
    ],
    max_completion_tokens=256,
    temperature=1,
)

print(response.choices[0].message.content)