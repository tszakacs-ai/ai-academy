def valida_prompt(prompt):
    """
    Controlla il prompt prima di inviarlo all'AI per prevenire injection attacks.
    
    Args:
        prompt (str): Il prompt dell'utente da validare
        
    Returns:
        bool: True se il prompt è valido
        
    Raises:
        ValueError: Se il prompt contiene parole vietate o è troppo lungo
    """
    # 1. Lista di parole/frasi da bloccare
    blacklist = [
        "ignora istruzioni",
        "resetta ruolo",
        "password",
        "dimentica",
        "bypassa",
        "istruzioni precedenti",
        "non rispettare",
        "viola",
        "dai precedenza",
        "credenziali",
        "ruolo originale",
        "disattiva",
        "sii un",
        "comportati come",
        "rivelami",
        "jailbreak"
    ]
    
    # 2. Controllo presenza parole vietate
    prompt_lower = prompt.lower()
    for parola in blacklist:
        # Controlla se la parola è presente nel prompt (case-insensitive)
        if parola.lower() in prompt_lower:
            raise ValueError(f"Prompt bloccato: contiene '{parola}'")
    
    # 3. Limite sulla lunghezza del prompt
    max_length = 400  # es: massimo 400 caratteri
    # Controlla se il prompt è troppo lungo
    if len(prompt) > max_length:
        raise ValueError(f"Prompt troppo lungo: {len(prompt)} caratteri (massimo {max_length})")
    
    # 4. Altri controlli (struttura, presenza variabili non consentite, ecc.)
    # Controllo presenza di delimitatori di sistema o caratteri sospetti
    suspicious_patterns = ["```", "<system>", "</system>", "<instructions>", "DAN"]
    for pattern in suspicious_patterns:
        if pattern in prompt:
            raise ValueError(f"Prompt bloccato: contiene pattern sospetto '{pattern}'")
            
    # Controllo del rapporto maiuscole/minuscole (possibile tentativo di evasione)
    uppercase_count = sum(1 for c in prompt if c.isupper())
    if len(prompt) > 0 and uppercase_count / len(prompt) > 0.5:
        raise ValueError("Prompt bloccato: troppe lettere maiuscole")

    # Se supera tutti i controlli
    return True

# Esempio d'uso
if __name__ == "__main__":
    prompt_utente = input("Inserisci il prompt da controllare: ")
    try:
        if valida_prompt(prompt_utente):
            print("Prompt accettato. Procedo con l'invio al modello.")
    except ValueError as e:
        print("Errore:", e)