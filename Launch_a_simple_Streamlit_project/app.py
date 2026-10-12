import streamlit as st

from views.calculator_view import calculator_page
from views.greeting_view import greeting_page
from views.login_view import login_page


def initialize_session_state():
    """Initialize required session state values."""

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "username" not in st.session_state:
        st.session_state.username = ""

    if "show_greeting" not in st.session_state:
        st.session_state.show_greeting = False


def main():
    st.set_page_config(
        page_title="Factorial Calculator",
        page_icon="🔢",
        layout="centered"
    )

    initialize_session_state()

    if st.session_state.logged_in:
        calculator_page()

    elif st.session_state.show_greeting:
        greeting_page()

    else:
        login_page()


if __name__ == "__main__":
    main()