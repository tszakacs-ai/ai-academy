import os

# Legge la chiave API dal file di segreto
try:
    with open('/run/secrets/api_key', 'r') as f:
        api_key = f.read().strip()
    print(f"Ciao! La tua chiave è: {api_key}")
except FileNotFoundError:
    # Fallback se il segreto non è disponibile
    api_key = os.getenv('API_KEY', 'chiave-non-trovata')
    print(f"Ciao! La tua chiave è: {api_key}")