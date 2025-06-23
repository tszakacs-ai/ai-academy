
def valida_prompt(prompt):
     # 0. Scrivi prompt base per il controllo
    """
    Controlla il prompt prima di inviarlo  al modello per prevenire
    possibili attacchi di prompt injection.
   
    """
    # 1. Lista di parole/frasi da bloccare
    blacklist = [
        "ignora istruzioni",
        "resetta ruolo",
        "password",
        "esegui codice",
        "cambia modello",   
        "modifica dati",
        "cancella tutto",
        "esegui comandi",   
    ]
    
    # 2. Controllo presenza parole vietate
    for parola in blacklist:
        # COMPLETA: controlla se la parola è presente nel prompt (case-insensitive)
        if parola in blacklist:
            raise ValueError(f"Prompt bloccato: contiene '{parola}'")
    
    # 3. (FACOLTATIVO) Limite sulla lunghezza del prompt
    max_length = 400  # es: massimo 400 caratteri
    # COMPLETA: controlla se il prompt è troppo lungo
    if len(prompt) > max_length:
        raise ValueError("Prompt troppo lungo")
    
    # 4. (FACOLTATIVO) Altri controlli (struttura, presenza variabili non consentite, ecc.)
    # Esempio: controlla se il prompt contiene variabili non consentite
    pattern_malicious = ["<system>", "script>"]
    for pattern in pattern_malicious:
        # COMPLETA: controlla se il pattern è presente nel prompt
        if pattern in prompt:
            raise ValueError(f"Prompt bloccato: contiene pattern '{pattern}'")
    # Se supera tutti i controlli
    return True

# Esempio d’uso (DA COMPLETARE NEI PUNTI CON '...')
prompt_utente = input("Inserisci il prompt da controllare: ")
try:
    if valida_prompt(prompt_utente):
        print("Prompt accettato. Procedo con l’invio al modello.")
except ValueError as e:
    print("Errore:", e)
