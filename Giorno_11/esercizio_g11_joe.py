from dotenv import load_dotenv
import os

# Specifica il percorso del file .env se non è nella root
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

api_key = os.getenv("API_KEY")
api_env = os.getenv("API_ENV")

if api_key is None or api_env is None:
    print("Attenzione: una o più variabili non sono state trovate nel file .env")
else:
    print(f"Valori delle variabili: {api_key} {api_env}")