import streamlit as st

from utils.factorial_utils import factorial


def calculator_page():
    st.title("🔢 Factorial Calculator")
    st.write(f"Hello, **{st.session_state.username}**!")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.show_greeting = False
        st.rerun()

    st.divider()

    number = st.number_input(
        "Enter a non-negative integer:",
        min_value=0,
        max_value=900,
        step=1
    )

    if st.button("Calculate", type="primary"):
        result = factorial(number)

        st.success(f"{number}! = {result}")
        st.caption(f"The result contains {len(str(result))} digits.")