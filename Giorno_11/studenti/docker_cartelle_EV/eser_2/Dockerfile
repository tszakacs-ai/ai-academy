FROM python:3.11-slim

WORKDIR /app

COPY main.py .
COPY .env .

RUN pip install --no-cache-dir python-dotenv

CMD ["python", "main.py"]
