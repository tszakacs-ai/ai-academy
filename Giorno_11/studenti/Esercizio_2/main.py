import os

# Carica la chiave API dalle variabili d'ambiente
api_key = os.getenv('API_KEY', 'chiave_non_trovata')

print(f"Ciao! La tua chiave è: {api_key}")