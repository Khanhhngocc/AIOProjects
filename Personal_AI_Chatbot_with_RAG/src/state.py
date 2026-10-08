# src/state.py

import time
import streamlit as st


def init_session_state():
    default_states = {
        "rag_chain": None,
        "models_loaded": False,
        "embeddings": None,
        "llm": None,

        # Chatbot version
        "chat_history": [],
        "pdf_processed": False,
        "pdf_name": "",
        "num_chunks": 0,
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_message(role, content):
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time(),
    })


def clear_chat():
    st.session_state.chat_history = []


def clear_document():
    st.session_state.rag_chain = None
    st.session_state.pdf_processed = False
    st.session_state.pdf_name = ""
    st.session_state.num_chunks = 0
    st.session_state.chat_history = []


def export_chat_history():
    lines = []

    for message in st.session_state.chat_history:
        role = (
            "USER"
            if message["role"] == "user"
            else "ASSISTANT"
        )

        lines.append(
            f"{role}:\n{message['content']}\n"
        )

    return "\n".join(lines)