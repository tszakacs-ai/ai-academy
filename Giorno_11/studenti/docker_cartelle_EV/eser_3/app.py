import os
from fastapi import FastAPI

app = FastAPI()

api_key = os.environ.get("OPENAI_API_KEY", "non impostata")
version = os.environ.get("VERSION", "0.0.1")

@app.get("/")
def greet_json():
    return {
        "message": "Hello World!",
        "api_key": api_key,
        "version": version
    }
