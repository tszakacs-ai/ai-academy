import time
import streamlit as st


def safe_gpt_call(func, *args, max_retries=5, wait_seconds=60, **kwargs):
    """Helper to retry Azure OpenAI calls when hitting rate limits."""
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
                raise e
    raise Exception("Superato il numero massimo di retry per il rate limit.")
