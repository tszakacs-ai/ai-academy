"""Streamlit entry point that delegates to ``main.py``.

Running ``streamlit run app.py`` will execute the ``main`` function
defined in ``main.py`` so that the full RAG demo starts when this file
is launched.
"""

from main import main


if __name__ == "__main__":
    main()
