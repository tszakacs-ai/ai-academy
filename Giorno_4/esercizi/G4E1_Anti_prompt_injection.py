def valida_prompt(prompt):
    """
    Controlla il prompt prima di inviarlo:
    - Blocca parole/frasi vietate
    - Controlla la lunghezza massima
    - (Facoltativo) Altri controlli
    """
    # 1. Lista di parole/frasi da bloccare
    blacklist = [
        "ignora istruzioni",
        "resetta ruolo",
        "password",
        "token",
        "bypass",
        "prompt injection",
        "esegui codice pericoloso",
        "comando nascosto",
        "sistema operativo",
        "shell",
        "sudo",
        "root"
    ]
    
    # 2. Controllo presenza parole vietate (case-insensitive)
    prompt_lower = prompt.lower()
    for parola in blacklist:
        if parola in prompt_lower:
            raise ValueError(f"Prompt bloccato: contiene '{parola}'")
    
    # 3. Limite sulla lunghezza del prompt
    max_length = 400  # es: massimo 400 caratteri
    if len(prompt) > max_length:
        raise ValueError("Prompt troppo lungo")
    
    if "${" in prompt:
        raise ValueError("Prompt contiene variabili non consentite (${...})")

    # Se supera tutti i controlli
    return True

# Esempio d’uso
prompt_utente = input("Inserisci il prompt da controllare: ")
try:
    if valida_prompt(prompt_utente):
        print("Prompt accettato. Procedo con l’invio al modello.")
except ValueError as e:
    print("Errore:", e)
