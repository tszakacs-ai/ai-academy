# Utilizza una immagine Python ufficiale leggera
FROM python:3.11-slim

# Imposta la directory di lavoro
WORKDIR /code

# Copia requirements e installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice sorgente
COPY . .

# Streamlit: imposta la porta e disabilita il controllo browser
ENV PORT 7860
ENV STREAMLIT_SERVER_PORT 7860
ENV STREAMLIT_SERVER_HEADLESS true
ENV STREAMLIT_SERVER_ENABLECORS false

# (Opzionale) Nascondi i segreti: NON copiarli nel container!
# Usa variabili d'ambiente o .env fornito da Hugging Face Spaces

# Comando di avvio
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]