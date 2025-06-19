
def valida_prompt(prompt):
     # 0. Scrivi prompt base per il controllo
    """
    Controlla il prompt prima di inviarlo ...
   
    """
    # 1. Lista di parole/frasi da bloccare
    blacklist = [
        "ignora istruzioni",
        "resetta ruolo",
        "password",
        "dimentica tutto",
        "come hackerare",
        "phishing",
        "malware",
        "codice malevolo",
        "accesso non autorizzato",
        "creare virus",
        "attacco ddos",
        "vulnerabilità sistema",
        "backdoor",
        "sql injection",
        "exploit",
        "botnet",
        "ransomware",
        "furto dati",
        "informazioni private",
        "carta di credito",
        "credenziali",
        "conto bancario",
        "documenti sensibili",
        "droga",
        "armi",
        "violenza",
        "odio",
        "discriminazione",
        "autolesionismo",
        "suicidio",
        "eludere sicurezza",
        "generare testo offensivo",
        "sviare controlli",
        "bypassare filtri"
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
    # AGGIUNGI ALTRI CONTROLLI QUI

    # Se supera tutti i controlli
    return True

# Esempio d’uso (DA COMPLETARE NEI PUNTI CON '...')
prompt_utente = input("Inserisci il prompt da controllare: ")
try:
    if valida_prompt(prompt_utente):
        print("Prompt accettato. Procedo con l’invio al modello.")
except ValueError as e:
    print("Errore:", e)
