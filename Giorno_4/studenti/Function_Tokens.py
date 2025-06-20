import tiktoken as tk
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Conta il numero di token in un testo per un modello specifico.
    
    Args:
        text (str): Il testo da analizzare.
        model (str): Il modello OpenAI da utilizzare per il conteggio dei token.
    
    Returns:
        int: Numero di token nel testo.
    """
    encoding = tk.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

if __name__ == "__main__":
    # Esempio d'uso
    testo = "Ciao, questo è un esempio di testo per contare i token."
    modello = "gpt-3.5-turbo"
    numero_token = count_tokens(testo, modello)
    print(f"Il testo contiene {numero_token} token per il modello {modello}.")