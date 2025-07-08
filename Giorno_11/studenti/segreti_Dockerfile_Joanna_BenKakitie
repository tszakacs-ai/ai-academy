FROM python:3.11-slim

# Installa la libreria per leggere .env
RUN pip install python-dotenv

# Copia gli script nel container
COPY main.py ./
COPY .env ./

# Esegui lo script
CMD ["python", "main.py"]
