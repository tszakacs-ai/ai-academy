from dotenv import load_dotenv
import os

load_dotenv()  # Carica le variabili da .env
api_key = os.getenv("MY_SECRET_KEY")

print(f"La chiave segreta è: {api_key}")
