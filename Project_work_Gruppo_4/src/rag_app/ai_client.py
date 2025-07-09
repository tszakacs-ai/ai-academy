import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import streamlit as st


class AIProjectClientDefinition:
    """Wrapper per inizializzare Azure AI Project client."""

    def __init__(self):
        endpoint = os.getenv("PROJECT_ENDPOINT")
        if not endpoint:
            st.error(
                "ERRORE: PROJECT_ENDPOINT non definito nel file .env. Assicurati che il file .env sia presente e configurato correttamente."
            )
            st.stop()
        self.endpoint = endpoint
        try:
            self.client = AIProjectClient(
                endpoint=self.endpoint,
                azure_endpoint=self.endpoint,
                credential=DefaultAzureCredential(),
            )
        except Exception as e:
            st.exception(
                f"ERRORE CRITICO: Impossibile inizializzare AIProjectClient. Dettagli: {e}"
            )
            st.stop()
