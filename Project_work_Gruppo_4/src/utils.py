import time
import streamlit as st


def safe_gpt_call(func, *args, max_retries=5, wait_seconds=60, **kwargs):
    """Retry helper for Azure OpenAI calls with basic error handling."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 429:
                st.warning(
                    f"Rate limit Azure superato! Aspetto {wait_seconds} secondi... (Tentativo {attempt+1}/{max_retries})"
                )
                time.sleep(wait_seconds)
            else:
                st.error(f"Errore durante la chiamata API di Azure OpenAI: {e}")
                return type('obj', (object,), {'choices': [{'message': {'content': f"Errore API: {e}"}}]})()
    st.error("Superato il numero massimo di retry per il rate limit.")
    return type('obj', (object,), {'choices': [{'message': {'content': "Errore: Superato il numero massimo di tentativi per il rate limit."}}]})()
