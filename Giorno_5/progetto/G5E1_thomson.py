# azure_openai_client.py

from openai import AzureOpenAI

# Configurazione
AZURE_OPENAI_ENDPOINT = "https://matth-mc4pfxah-swedencentral.cognitiveservices.azure.com/"
AZURE_OPENAI_API_KEY = "2UmwCov9HxZFqplz7tKqP9IoZzeiyLPrY1TfXNAKVt7G9HgWDcx1JQQJ99BFACfhMk5XJ3w3AAAAACOGPsza" 
AZURE_OPENAI_DEPLOYMENT_NAME = "chatgpt-thomson3"

# Inizializza client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-12-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Funzione generica
def call_azure_openai_chat(
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int = 1000,
    temperature: float = 1.0,
    top_p: float = 1.0,
    deployment_name: str = AZURE_OPENAI_DEPLOYMENT_NAME
) -> str:
    """
    Invoca Azure OpenAI Chat completions.

    Args:
        system_prompt (str): ruolo system
        user_prompt (str): contenuto utente
        max_completion_tokens (int): max token per la risposta
        temperature (float): creatività
        top_p (float): nucleus sampling
        deployment_name (str): deployment name su Azure

    Returns:
        str: risposta generata
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return response.choices[0].message.content

# Funzione per leggere documento e costruire prompt
def process_document_and_ask(
    document_path: str,
    system_prompt: str,
    instruction_prompt: str,
    max_completion_tokens: int = 1000,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> str:
    """
    Legge documento da file e invoca il modello.

    Args:
        document_path (str): path al file documento (txt)
        system_prompt (str): system prompt
        instruction_prompt (str): istruzioni per l'utente
        max_completion_tokens (int), temperature (float), top_p (float): parametri modello

    Returns:
        str: risposta generata
    """
    # Leggi documento
    with open(document_path, "r", encoding="utf-8") as file:
        testo_ripulito = file.read()

    # Costruisci prompt utente
    user_prompt = f"""
    DOCUMENTO:

    {testo_ripulito}

    ISTRUZIONI:

    {instruction_prompt}
    """

    # Chiamata al modello
    return call_azure_openai_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p
    )

# Esempio di utilizzo
if __name__ == "__main__":
    risposta = process_document_and_ask(
        document_path="documento_test.txt",
        system_prompt="Sei un assistente aziendale. Il tuo compito è riassumere e analizzare il contenuto fornito, e generare una risposta adeguata per il cliente.",
        instruction_prompt="""
        1. Fornisci un riepilogo formale del contenuto (summary).
        2. Indica i punti salienti o eventuali criticità.
        3. Genera una bozza di risposta da inviare al cliente, in tono cortese e professionale.
        """,
        max_completion_tokens=1000,
        temperature=1.0
    )

    print("\n--- Risposta completa generata ---\n")
    print(risposta)
