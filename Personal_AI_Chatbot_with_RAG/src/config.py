# src/config.py

EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 256

RETRIEVER_K = 4

SEMANTIC_CHUNK_CONFIG = {
    "buffer_size": 1,
    "breakpoint_threshold_type": "percentile",
    "breakpoint_threshold_amount": 95,
    "min_chunk_size": 500,
    "add_start_index": True,
}

RAG_PROMPT = """
You are a document question-answering assistant.

Answer the question using only the information provided in the Context.

Rules:
- Do not make up information that is not present in the Context.
- If the Context does not contain enough information, say:
  "I do not find information in this Document."
- Answer clearly in Vietnamese.

Context:
{context}

Question:
{question}

Answer:
"""