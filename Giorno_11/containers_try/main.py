import os
from dotenv import load_dotenv

load_dotenv()
print(f"La tua chiave è {os.getenv('Api_key')}")