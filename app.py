"""Root Streamlit entry point (preferred): ``streamlit run app.py``."""

from __future__ import annotations

from src.app_streamlit import main

if __name__ == "__main__":
    main()
else:
    # Streamlit executes the script as a module body (not __main__).
    main()
