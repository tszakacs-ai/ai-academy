import re

text = "Mario Rossi ha ricevuto un bonifico sull'IBAN IT60X0542811101000000123456, dall'IBAN IT60X054281110100A000678456"

iban_pattern = r"\bIT\d{2}[A-Z0-9]{1,23}\b"

ibans = re.findall(iban_pattern, text)

for iban in ibans:
    print("IBAN:", iban)
