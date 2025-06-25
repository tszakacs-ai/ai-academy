def valida_prompt(prompt):
    # 0. Scrivi prompt base per il controllo
    """
    Controlla il prompt prima di inviarlo, bloccando contenuti vietati o strutture non ammesse.
    """

    # 1. Lista di parole/frasi da bloccare
    blacklist = [
        "ignora istruzioni",
        "resetta ruolo",
        "password",
        "token di accesso",
        "bypass",
        "admin",
        "comando shell",
        "inietta codice",
        "hacking"
        # AGGIUNGI ALTRE PAROLE CHIAVE QUI
    ]

    # 2. Controllo presenza parole vietate (case-insensitive)
    prompt_lower = prompt.lower()
    for parola in blacklist:
        if parola.lower() in prompt_lower:
            raise ValueError(f"Prompt bloccato: contiene '{parola}'")

    # 3. (FACOLTATIVO) Limite sulla lunghezza del prompt
    max_length = 400  # es: massimo 400 caratteri
    if len(prompt) > max_length:
        raise ValueError("Prompt troppo lungo")

    # 4. (FACOLTATIVO) Altri controlli (es. presenza di variabili vietate)
    # Ad esempio, vietiamo l'uso di $$VAR$$ come pattern sospetto
    if "$$" in prompt:
        raise ValueError("Prompt contiene pattern non consentiti ('$$')")

    # Se supera tutti i controlli
    return True


# Esempio d’uso
prompt_utente = input("Inserisci il prompt da controllare: ")
try:
    if valida_prompt(prompt_utente):
        print("✅ Prompt accettato. Procedo con l’invio al modello.")
except ValueError as e:
    print("❌ Errore:", e)




# Calcolo del costo in base al numero di token per modello

import tiktoken

# Tariffe per token (USD/token), già convertite da prezzo per milione
tariffe_token = {
    "gpt-4.1": {
        "ingresso": 0.000002,
        "uscita": 0.000008,
    },
    "gpt-4.1-mini": {
        "ingresso": 0.0000004,
        "uscita": 0.0000016,
    },
    "gpt-4.1-nano": {
        "ingresso": 0.0000001,
        "uscita": 0.0000004,
    },
    "openai-o3": {
        "ingresso": 0.000002,
        "uscita": 0.000008,
    },
    "openai-o4-mini": {
        "ingresso": 0.0000011,
        "uscita": 0.0000044,
    },
}

def conta_token(testo: str, modello: str) -> int:
    """
    Restituisce il numero di token presenti in un testo, in base al modello specificato.
    """
    try:
        codifica = tiktoken.encoding_for_model(modello)
    except KeyError:
        codifica = tiktoken.get_encoding("cl100k_base")
    return len(codifica.encode(testo))

def stima_costo(testo: str, modello: str, token_output_previsti: int = 0):
    """
    Restituisce una stima del costo totale (input + output) per il modello scelto.
    """
    modello = modello.lower()
    if modello not in tariffe_token:
        raise ValueError(f"Modello non riconosciuto: '{modello}'")

    token_input = conta_token(testo, modello)
    tariffa_ingresso = tariffe_token[modello]["ingresso"]
    tariffa_uscita = tariffe_token[modello]["uscita"]

    costo_ingresso = token_input * tariffa_ingresso
    costo_uscita = token_output_previsti * tariffa_uscita
    costo_complessivo = costo_ingresso + costo_uscita

    return {
        "token_input": token_input,
        "costo_input": costo_ingresso,
        "token_output": token_output_previsti,
        "costo_output": costo_uscita,
        "totale": costo_complessivo
    }

# Interfaccia principale
if __name__ == "__main__":
    modello_selezionato = input("Seleziona un modello (es: gpt-4.1, openai-o4-mini, ecc.): ").strip().lower()
    contenuto = input("Inserisci il testo del prompt: ")

    try:
        stimati_output = input("Stima token output (puoi lasciare vuoto): ")
        token_output = int(stimati_output) if stimati_output.strip() else 0

        risultato = stima_costo(contenuto, modello_selezionato, token_output)

        print("\n--- Riepilogo Stima ---")
        print(f"Token in input: {risultato['token_input']}")
        print(f"Costo input: ${risultato['costo_input']:.8f}")
        print(f"Token output stimati: {risultato['token_output']}")
        print(f"Costo output: ${risultato['costo_output']:.8f}")
        print(f"Costo totale stimato: ${risultato['totale']:.8f}")

    except ValueError as err:
        print("⚠️ Errore:", err)
