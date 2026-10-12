# PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application for question answering over PDF documents, with a retrieval pipeline primarily optimized for Vietnamese content.
The application can also handle English documents and queries in basic tests, although English retrieval quality is not guaranteed to match Vietnamese performance.

The project combines semantic document chunking, vector search, and a local Hugging Face language model to generate answers grounded in the uploaded document.

Two Streamlit applications are included:

- **Basic RAG App** — a simple interface for understanding the core RAG workflow.
- **RAG Chatbot App** — an extended chatbot interface with conversation history and document management.

## Live Demo

[Please, be curious and try the chatbot version!](https://pdf-rag-chatbot-with-me.streamlit.app/)

[Please, be curious and try the basic version!](https://pdf-rag-basic.streamlit.app/)



---

## Features

### Basic RAG App

- Upload PDF documents
- Extract text from PDF files
- Split documents using semantic chunking
- Generate embeddings using a Vietnamese bi-encoder
- Store document embeddings in Chroma
- Retrieve relevant document chunks
- Generate answers using a local Hugging Face LLM
- Simple Streamlit question-answering interface

### RAG Chatbot App

The chatbot version extends the basic RAG application with:

- Chat-style interface using Streamlit
- Conversation history
- PDF processing status
- Current document information
- Number of generated chunks
- Clear chat history
- Remove the current document
- Export chat history as a text file
- Error handling for PDF processing and answer generation
- Sidebar control panel
- Disabled chat input until a document is ready

---

## Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **Hugging Face Transformers**
* **Sentence Transformers**
* **ChromaDB**
* **PyTorch**
* **PyPDF**
* **Qwen2.5**

---

## Models

### Embedding Model

```text
bkai-foundation-models/vietnamese-bi-encoder
```

The embedding model converts document chunks and user queries into vector representations for semantic similarity search.

### Language Model

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model runs locally through the Hugging Face `transformers` pipeline.

On supported Apple Silicon devices, the application uses the PyTorch `MPS` backend. Otherwise, it falls back to CPU execution.

---

## Running the Application

### Basic RAG Application

Run:

```bash
streamlit run rag_app.py
```

This version is useful for understanding the basic RAG workflow:


### RAG Chatbot Application

Run:

```bash
streamlit run rag_chatbot_app.py
```

The chatbot version provides a more complete user experience with chat history, document status, document removal, and conversation export.

Streamlit will normally start the application at:

```text
http://localhost:8501
```

---

## How to Use

1. Start either the Basic RAG App or the RAG Chatbot App.
2. Wait for the embedding model and language model to load.
3. Upload a PDF document.
4. Click **Process PDF**.
5. The document is loaded and divided into semantic chunks.
6. The chunks are converted into embeddings and stored in Chroma.
7. Enter a question about the document.
8. The retriever selects the most relevant chunks.
9. The retrieved context and question are passed to the language model.
10. The generated answer is displayed in the interface.

---

## How the RAG System Works

### 1. PDF Loading

`PyPDFLoader` loads the uploaded PDF and converts its pages into LangChain `Document` objects.

### 2. Semantic Chunking

Instead of splitting text only by character count, the application uses `SemanticChunker`.

The splitter uses embeddings to detect semantic changes in the document and creates chunks based on changes in meaning.

### 3. Embedding

Each chunk is converted into a numerical vector using:

```text
bkai-foundation-models/vietnamese-bi-encoder
```

These vectors represent the semantic meaning of the text.

### 4. Vector Database

The document vectors are stored in Chroma.

When the user asks a question, the question is also converted into an embedding.

### 5. Retrieval

The retriever compares the question vector with document vectors and returns the most semantically relevant chunks.

The current configuration retrieves the top:

```text
k = 4
```

chunks.

### 6. Generation

The retrieved chunks are inserted into a prompt together with the user's question.

The local Qwen model then generates an answer based on the retrieved context.

---

## Configuration

The main configuration is located in:

```text
src/config.py
```

Example:

```python
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 256

RETRIEVER_K = 4
```

This makes it easy to experiment with different embedding models, language models, generation limits, or retrieval settings without modifying the main application code.

---


## Current Limitations

* Only PDF documents are currently supported.
* The vector database is created in memory and is not persisted between application sessions.
* Only one processed document is used at a time.
* The application uses a relatively small local language model to keep hardware requirements manageable.
* Chat history is currently used for the user interface only.
* Previous messages are not yet incorporated into retrieval or question rewriting.
* The quality of generated answers depends on the quality of PDF text extraction and retrieved chunks.

---

## Possible Future Improvements

Potential improvements include:

* Multi-PDF support
* Persistent Chroma collections
* Source citation and page-number display
* Retrieval score visualization
* Conversational memory
* History-aware question rewriting
* Hybrid search
* Reranking
* Support for DOCX and TXT files
* Streaming responses
* Multiple LLM options
* Configurable retrieval parameters
* Evaluation of retrieval and answer quality

---

## Learning Objectives

This project was built to practice and understand the main components of a practical RAG system:

* Document loading
* Document chunking
* Text embeddings
* Vector databases
* Semantic retrieval
* Prompt construction
* Local LLM inference
* LangChain Expression Language (LCEL)
* Streamlit application development
* Modular Python project organization

---
