import streamlit as st


def greeting_page():
    st.title("Hello!")

    st.write(f"Hello, **{st.session_state.username}**!")
    st.info("You do not have permission to access the factorial calculator.")

    if st.button("Back to Login"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.show_greeting = False
        st.rerun()