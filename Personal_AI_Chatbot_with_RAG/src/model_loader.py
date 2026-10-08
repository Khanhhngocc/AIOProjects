# src/model_loader.py

import torch
import streamlit as st

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from src.config import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    MAX_NEW_TOKENS,
)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )


@st.cache_resource
def load_llm():
    # Apple Silicon → MPS
    # Máy khác → CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model_dtype = torch.float16
    else:
        device = torch.device("cpu")
        model_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )

    model = model.to(device)
    model.eval()

    model_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    return HuggingFacePipeline(
        pipeline=model_pipeline
    )