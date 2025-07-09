"""Streamlit entry point for Hugging Face Spaces.

This tiny wrapper imports the RAG demo contained in ``Giorno_11``
and runs its ``main`` function. Hugging Face automatically runs
``app.py`` from the repository root, so this file just forwards the
execution.
"""

from src.main import main

if __name__ == "__main__":
    main()
