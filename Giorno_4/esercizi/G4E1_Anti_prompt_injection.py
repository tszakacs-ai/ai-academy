def valida_prompt(prompt):
    """
    Controlla il prompt prima di inviarlo al modello per prevenire injection o input non validi.

    Parameters
    ----------
    prompt : str
        Il prompt fornito dall'utente.

    Returns
    -------
    bool
        True se il prompt è valido, altrimenti solleva un'eccezione.
    """
    # 1. Lista di parole/frasi da bloccare
    blacklist = [
        "ignora istruzioni",
        "resetta ruolo",
        "password",
        "elimina dati",
        "bypass",
        "admin",
        "sudo",
        "hack",
        "override",
        "token",
        "chiave segreta",
        "credenziali"
        # AGGIUNGI ALTRE PAROLE CHIAVE QUI
    ]
    
    # 2. Controllo presenza parole vietate
    for parola in blacklist:
        # Controlla se la parola è presente nel prompt (case-insensitive)
        if parola.lower() in prompt.lower():
            raise ValueError(f"Prompt bloccato: contiene '{parola}'")
    
    # 3. (FACOLTATIVO) Limite sulla lunghezza del prompt
    max_length = 400  # es: massimo 400 caratteri
    # Controlla se il prompt è troppo lungo
    if len(prompt) > max_length:
        raise ValueError("Prompt troppo lungo")
    
    # 4. (FACOLTATIVO) Altri controlli (struttura, presenza variabili non consentite, ecc.)
    # Esempio: verifica che il prompt non contenga solo spazi o sia vuoto
    if not prompt.strip():
        raise ValueError("Prompt vuoto o composto solo da spazi")
    
    # Se supera tutti i controlli
    return True

# Esempio d’uso
prompt_utente = input("Inserisci il prompt da controllare: ")
try:
    if valida_prompt(prompt_utente):
        print("Prompt accettato. Procedo con l’invio al modello.")
except ValueError as e:
    print("Errore:", e)
