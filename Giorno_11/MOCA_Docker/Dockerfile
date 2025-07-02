FROM python:3.10-slim

WORKDIR /app

# Copia requirements.txt e installa le dipendenze
COPY Giorno_11/MOCA_Docker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# Copia tutto il progetto nella directory /app
COPY . .

# Copia il file .env nella directory di lavoro finale (dove c'è app.py)
COPY Giorno_11/MOCA_Docker/.env /app/Giorno_11/MOCA_Docker/.env

WORKDIR /app/Giorno_11/MOCA_Docker

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]