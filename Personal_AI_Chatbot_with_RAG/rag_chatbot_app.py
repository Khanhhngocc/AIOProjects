# rag_chatbot_app.py

from pathlib import Path

import streamlit as st

from src.state import (
    init_session_state,
    add_message,
    clear_chat,
    clear_document,
    export_chat_history,
)
from src.model_loader import load_embeddings, load_llm
from src.pdf_processor import process_pdf
from src.rag_service import build_rag_chain


def load_models():
    if st.session_state.models_loaded:
        return

    with st.spinner("Loading AI models..."):
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True


def process_uploaded_pdf(uploaded_file):
    docs = process_pdf(uploaded_file, st.session_state.embeddings)

    st.session_state.rag_chain = build_rag_chain(
        docs=docs,
        embeddings=st.session_state.embeddings,
        llm=st.session_state.llm,
    )

    st.session_state.pdf_processed = True
    st.session_state.pdf_name = uploaded_file.name
    st.session_state.num_chunks = len(docs)

    clear_chat()

    add_message(
        "assistant",
        f"Successfully processed **{uploaded_file.name}**.\n\n"
        f"The document was split into **{len(docs)} chunks**. "
        f"You can now start asking questions."
    )


def display_chat():
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.write(
                "Hello! I'm your PDF RAG Assistant. "
                "Upload a PDF document to get started."
            )
        return

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def generate_answer(question):
    output = st.session_state.rag_chain.invoke(question)

    if "Answer:" in output:
        return output.split("Answer:", 1)[1].strip()

    return output.strip()


def main():
    st.set_page_config(
        page_title="PDF RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    logo_path = Path("logo.png")
    if logo_path.exists():
        st.logo(str(logo_path), size="large")

    # SIDEBAR
    with st.sidebar:
        st.title("Control Panel")

        st.subheader("AI Model")

        if not st.session_state.models_loaded:
            st.warning("Models are loading...")
            load_models()
            st.success("Models are ready!")
            st.rerun()
        else:
            st.success("Models are ready")

        st.divider()

        st.subheader("Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF to use as the chatbot's knowledge base.",
        )

        if uploaded_file and st.button(
            "Process PDF",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Processing document..."):
                try:
                    process_uploaded_pdf(uploaded_file)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")

        if st.session_state.pdf_processed:
            st.success("Document is ready")
            st.write(f"**{st.session_state.pdf_name}**")
            st.caption(f"{st.session_state.num_chunks} chunks")

            if st.button("Remove document", use_container_width=True):
                clear_document()
                st.rerun()
        else:
            st.info("No document has been processed yet.")

        st.divider()

        st.subheader("Chat")

        if st.button(
            "Clear chat history",
            use_container_width=True,
            disabled=not st.session_state.chat_history,
        ):
            clear_chat()
            st.rerun()

        if st.session_state.chat_history:
            st.download_button(
                "Download chat history",
                data=export_chat_history(),
                file_name="rag_chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.divider()

        with st.expander("How to use"):
            st.markdown("""
            1. Upload a PDF file.
            2. Click **Process PDF**.
            3. Wait for the system to process the document.
            4. Enter your question in the chat box.
            5. The chatbot will answer based on the PDF content.
            """)

    # MAIN CONTENT
    st.title("PDF RAG Chatbot")
    st.caption("Chat with your PDF documents using Retrieval-Augmented Generation")

    if st.session_state.pdf_processed:
        st.info(
            f"Currently using: **{st.session_state.pdf_name}** "
            f"· {st.session_state.num_chunks} chunks"
        )

    display_chat()

    # CHAT INPUT
    if not st.session_state.pdf_processed:
        st.info("Upload and process a PDF from the sidebar first.")
        st.chat_input("Please upload a PDF first...", disabled=True)
        return

    user_input = st.chat_input("Ask a question about the document...")

    if user_input:
        add_message("user", user_input)

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching for relevant information..."):
                try:
                    answer = generate_answer(user_input)
                    st.write(answer)
                    add_message("assistant", answer)

                except Exception as e:
                    error_message = "Sorry, an error occurred while generating the answer."
                    st.error(error_message)

                    with st.expander("Error details"):
                        st.exception(e)

                    add_message("assistant", error_message)


if __name__ == "__main__":
    main()