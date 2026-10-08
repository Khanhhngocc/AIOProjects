# src/rag_service.py

from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import (
    RAG_PROMPT,
    RETRIEVER_K,
)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def build_rag_chain(
    docs,
    embeddings,
    llm,
):
    # Vector database.
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
    )

    # Semantic retriever.
    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": RETRIEVER_K
        }
    )

    # Prompt cho LLM.
    prompt = PromptTemplate.from_template(
        RAG_PROMPT
    )

    # RAG pipeline.
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain