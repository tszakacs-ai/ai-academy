import os

api_key = os.getenv("MY_API_KEY")

if not api_key:
    raise ValueError("❌ API Key mancante!")

print("✅ La tua API key è:", api_key[:5] + "..." if api_key else "Nessuna")