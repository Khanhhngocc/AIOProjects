from pathlib import Path

import streamlit as st


PROJECT_DIR = Path(__file__).parent.parent
USER_FILE = PROJECT_DIR / "data" / "users.txt"


def load_users():
    """Read and return the list of allowed users."""

    try:
        if not USER_FILE.exists():
            st.error(f"User file not found: {USER_FILE}")
            return []

        with open(USER_FILE, "r", encoding="utf-8") as file:
            users = [
                line.strip()
                for line in file.readlines()
                if line.strip()
            ]

        return users

    except Exception as error:
        st.error(f"Error reading the user file: {error}")
        return []