import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


# Carica variabili da .env nella stessa directory
load_dotenv()

def saluta():
    api_key = os.environ.get("API_KEY")
    if api_key:
        st.success(f"Ciao! La tua chiave è: {api_key}")
    else:
        st.warning("Ciao! Nessuna chiave fornita.")

if __name__ == "__main__":
    st.title("Benvenuto in Streamlit!")
    saluta()