# rag_app.py

import streamlit as st

from src.state import init_session_state
from src.model_loader import load_embeddings, load_llm
from src.pdf_processor import process_pdf
from src.rag_service import build_rag_chain


def load_models():
    if st.session_state.models_loaded:
        return

    with st.spinner("Loading models..."):
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True


def main():
    st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
    init_session_state()

    st.title("PDF RAG Assistant")
    st.markdown("""
    **An AI application that allows you to ask questions directly about the content of PDF documents.**

    **How to use:**
    1. **Upload PDF** → Select a PDF file and click **Process PDF**
    2. **Ask a question** → Enter a question about the document
    3. The AI will retrieve relevant information from the PDF and generate an answer

    ---
    """)

    # LOAD MODELS
    if not st.session_state.models_loaded:
        st.info("Loading models...")
        load_models()
        st.success("Models are ready!")
        st.rerun()

    # UPLOAD PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            try:
                docs = process_pdf(uploaded_file, st.session_state.embeddings)

                st.session_state.rag_chain = build_rag_chain(
                    docs=docs,
                    embeddings=st.session_state.embeddings,
                    llm=st.session_state.llm,
                )

                st.success(f"Done! The document was split into {len(docs)} chunks.")

            except Exception as e:
                st.error(f"Failed to process PDF: {e}")

    # QUESTION & ANSWER
    if st.session_state.rag_chain:
        st.markdown("---")
        question = st.text_input("Ask a question:")

        if question:
            with st.spinner("Generating answer..."):
                try:
                    output = st.session_state.rag_chain.invoke(question)
                    answer = output.split("Answer:", 1)[1].strip() if "Answer:" in output else output.strip()

                    st.write("**Answer:**")
                    st.write(answer)

                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")


if __name__ == "__main__":
    main()