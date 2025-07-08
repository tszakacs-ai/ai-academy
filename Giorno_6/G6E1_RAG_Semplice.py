import openai

def load_document(path):
    """Load the content of a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(document_text):
    """
    Build a prompt for RAG classification.
    """
    return (
        "Sei un assistente AI. "
        "Usa solo il testo fornito per rispondere. "
        "Determina il tipo di documento tra: mail, nota di credito, ordine di acquisto, contratto, altro.\n\n"
        "Testo del documento:\n"
        f"{document_text}\n\n"
        "Rispondi solo con la categoria del documento."
    )

# Choose the document to test
document_path = "Giorno_6/mail.txt"  # Change to 'nota_credito.txt', 'ordine_acquisto.txt' or 'contratto.txt' to test other cases
document_text = load_document(document_path)
prompt = build_prompt(document_text)

# Create client Azure OpenAI
client = openai.AzureOpenAI(
    api_key="6yqJB9WwC4PNOSMt9G1nczmbWluvV7T59emz6dYQxBAQc8hVfCNxJQQJ99BFACqBBLyXJ3w3AAAAACOGSzvA",  # <-- La tua chiave API di Azure OpenAI
    azure_endpoint="https://danie-mc4s81np-southeastasia.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview", # <-- Il tuo  API endpoint di Azure OpenAI
    api_version="2025-01-01-preview",
)

response = client.chat.completions.create(
    model="o4-mini",
    messages=[
        {"role": "system", "content": "Sei un assistente AI."},
        {"role": "user", "content": prompt}
    ],
    max_completion_tokens=1024,
    temperature=1,
)

print("Risposta:", response.choices[0].message.content)