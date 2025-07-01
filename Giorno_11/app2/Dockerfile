FROM python:3.12-slim
WORKDIR /app2
COPY main.py .
COPY .env .
RUN pip install python-dotenv
CMD ["python", "main.py"]
