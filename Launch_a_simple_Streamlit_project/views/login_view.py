import streamlit as st

from services.user_service import load_users

def login_page():
    st.title("Login")
    st.write("Enter your username to access the factorial calculator.")

    username = st.text_input(
        "Username:",
        placeholder="For example: huyen"
    )

    if st.button("Login", type="primary"):
        username = username.strip()

        if not username:
            st.warning("Please enter your username.")
            return

        users = load_users()

        if username in users:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.show_greeting = False
            st.rerun()

        else:
            st.session_state.logged_in = False
            st.session_state.username = username
            st.session_state.show_greeting = True
            st.rerun()