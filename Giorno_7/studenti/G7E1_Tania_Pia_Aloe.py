import re

# Testo di esempio
text = "Mario Rossi ha ricevuto un bonifico sull’IBAN IT60X0542811101000000123456. La sua email è mario.rossi@example.com."

# Liste di regex + etichette
patterns = [
    (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NOME]'),  # Nome Cognome
    (r'\bIT\d{2}[A-Z]\d{10,30}\b', '[IBAN]'),    # IBAN 
    (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[EMAIL]'),  # Email
]

# Anonimizzazione
for pattern, label in patterns:
    text = re.sub(pattern, label, text)

# Risultato
print(text)
