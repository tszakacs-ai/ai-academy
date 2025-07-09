def valida_prompt(prompt):
    """
    Controlla il prompt prima di inviarlo per evitare prompt injection.
    """
    # 1. Lista di parole/frasi da bloccare
    blacklist = [
        "ignora istruzioni",
        "resetta ruolo",
        "password",
        "hack",
        "exploit",
        "admin",
        "token",
        "chiave",
        "backdoor",
        "bypass",
        "override",
    ]
    
    # 2. Controllo presenza parole vietate (case-insensitive)
    prompt_lower = prompt.lower()
    for parola in blacklist:
        if parola.lower() in prompt_lower:
            raise ValueError(f"Prompt bloccato: contiene '{parola}'")
    
    # 3. Limite sulla lunghezza del prompt
    max_length = 400  # massimo 400 caratteri
    if len(prompt) > max_length:
        raise ValueError("Prompt troppo lungo")
    
    # 4. Altri controlli (facoltativi) da aggiungere qui

    # Se supera tutti i controlli
    return True


# Esempio d’uso
prompt_utente = input("Inserisci il prompt da controllare: ")
try:
    if valida_prompt(prompt_utente):
        print("Prompt accettato. Procedo con l’invio al modello.")
except ValueError as e:
    print("Errore:", e)


# Esercizio STIMA DEL COSTO TOKEN-PER-TOKEN

import tiktoken

# Prezzi per token in USD (già convertiti da $/milione di token)
prezzi_modelli = {
    "gpt-4.1": {
        "input": 0.000002,   # 2.00 USD / 1.000.000 token
        "output": 0.000008,  # 8.00 USD / 1.000.000 token
    },
    "gpt-4.1-mini": {
        "input": 0.0000004,  # 0.40 USD / 1.000.000 token
        "output": 0.0000016, # 1.60 USD / 1.000.000 token
    },
    "gpt-4.1-nano": {
        "input": 0.0000001,  # 0.10 USD / 1.000.000 token
        "output": 0.0000004, # 0.40 USD / 1.000.000 token
    },
    "openai-o3": {
        "input": 0.000002,   # 2.00 USD / 1.000.000 token
        "output": 0.000008,  # 8.00 USD / 1.000.000 token
    },
    "openai-o4-mini": {
        "input": 0.0000011,  # 1.10 USD / 1.000.000 token
        "output": 0.0000044, # 4.40 USD / 1.000.000 token
    },
}

def calcola_token(prompt: str, modello: str) -> int:
    """
    Conta il numero di token del prompt usando tiktoken.
    """
    try:
        encoding = tiktoken.encoding_for_model(modello)
    except KeyError:
        # Se modello non riconosciuto, usa encoding di default cl100k_base
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(prompt)
    return len(tokens)

def calcola_costo(prompt: str, modello: str, output_tokens_stimati: int = 0):
    """
    Calcola il costo stimato in USD per input e output dati un prompt e modello.
    """
    modello = modello.lower()
    if modello not in prezzi_modelli:
        raise ValueError(f"Modello '{modello}' non supportato.")
    
    num_token_input = calcola_token(prompt, modello)
    
    prezzo_input = prezzi_modelli[modello]["input"]
    prezzo_output = prezzi_modelli[modello]["output"]
    
    costo_input = num_token_input * prezzo_input
    costo_output = output_tokens_stimati * prezzo_output
    costo_totale = costo_input + costo_output
    
    return num_token_input, costo_input, output_tokens_stimati, costo_output, costo_totale

if __name__ == "__main__":
    modello = input("Inserisci il modello (es: gpt-4.1, gpt-4.1-mini, openai-o3): ").strip().lower()
    prompt = input("Inserisci il prompt: ")
    try:
        output_tokens = int(input("Inserisci il numero stimato di token in output (default 0): ") or 0)
        num_token_input, costo_input, out_token, costo_output, costo_totale = calcola_costo(prompt, modello, output_tokens)
        print(f"\nToken input: {num_token_input}")
        print(f"Costo input: ${costo_input:.8f} USD")
        print(f"Token output stimati: {out_token}")
        print(f"Costo output stimato: ${costo_output:.8f} USD")
        print(f"Costo totale stimato: ${costo_totale:.8f} USD")
    except ValueError as e:
        print("Errore:", e)
