import re

def valida_prompt(prompt):
     # 0. Scrivi prompt base per il controllo
    if prompt is None or prompt.strip() == "":
        raise ValueError("Prompt non può essere vuoto")

    # 1. Lista di parole/frasi da bloccare
    blacklist = [
        "ignora istruzioni",
        "resetta ruolo",
        "password",
        "scaricare film",
        "hack",
        "pornografia"
    ]
    
    # 2. Controllo presenza parole vietate
    for parola in blacklist:
        # COMPLETA: controlla se la parola è presente nel prompt (case-insensitive)
        if parola in prompt.lower():
            raise ValueError(f"Prompt bloccato: contiene '{parola}'")
    
    # 3. (FACOLTATIVO) Limite sulla lunghezza del prompt
    max_length = 400  # es: massimo 400 caratteri
    # COMPLETA: controlla se il prompt è troppo lungo
    if len(prompt) > max_length:
        raise ValueError("Prompt troppo lungo")
    
    # 4. (FACOLTATIVO) Altri controlli (struttura, presenza variabili non consentite, ecc.)
    injection_patterns = [
        "execute", "exec", "esegui", "run",
        "sovrascrivi", "overwrite", "delete", "elimina",
        "rm ", "del ", "format", "formatta",
        "system", "shell", "cmd", "powershell"
    ]

    suspicious_commands = [p for p in injection_patterns if p in prompt.lower()]
    
    if suspicious_commands:
        critical_paths = [
            r'[/\\](?:bin|sbin|usr|etc|system32|windows)[/\\]',  # Directory di sistema
            r'[/\\](?:\.\.?[/\\]){2,}',  # Directory traversal
            r'[a-zA-Z]:[/\\](?:windows|program files|users)[/\\]',  # Path Windows critici
            r'[/\\](?:home|root|var|tmp)[/\\]\w+',  # Path Unix critici
        ]

        for pattern in critical_paths:
                if re.search(pattern, prompt, re.IGNORECASE):
                    raise ValueError(f"Prompt bloccato: rilevata possibile comando di injection")

    # Se supera tutti i controlli
    return True

# Esempio d’uso (DA COMPLETARE NEI PUNTI CON '...')
prompt_utente = input("Inserisci il prompt da controllare: ")
try:
    if valida_prompt(prompt_utente):
        print("Prompt accettato. Procedo con l’invio al modello.")
except ValueError as e:
    print("Errore:", e)
