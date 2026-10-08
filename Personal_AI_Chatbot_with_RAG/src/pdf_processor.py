# src/pdf_processor.py

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from src.config import SEMANTIC_CHUNK_CONFIG


def process_pdf(uploaded_file, embeddings):
    tmp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load PDF thành Document objects.
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Chia document dựa trên sự thay đổi semantic.
        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            **SEMANTIC_CHUNK_CONFIG,
        )

        docs = semantic_splitter.split_documents(
            documents
        )

        return docs

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)