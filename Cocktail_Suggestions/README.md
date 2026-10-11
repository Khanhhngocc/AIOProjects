# Cocktail Suggestion System

An AI-powered cocktail recommendation system built with **Python**, **Sentence Transformers**, **PostgreSQL**, **pgvector**, and **Streamlit**.

The application converts cocktail information and user preferences into vector embeddings and uses semantic similarity search to recommend cocktails based on ingredients, style, occasion, category, or mixed preferences.

---

## Features

* Search cocktails by name
* Recommend cocktails by ingredients
* Recommend cocktails by style or mood
* Recommend cocktails by occasion
* Combine multiple preferences in a single search
* Browse cocktails by category
* Discover random cocktails
* Semantic search using vector embeddings
* PostgreSQL vector search with pgvector
* Interactive Streamlit web interface
* CLI interface for simple terminal-based testing

---

## How It Works

The system uses the pretrained Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

to convert cocktail descriptions and user queries into 384-dimensional embedding vectors.

### Data preparation flow

```text
Cocktail Dataset
      ↓
Data Cleaning
      ↓
Combined Cocktail Text
      ↓
SentenceTransformer
      ↓
384-Dimensional Embeddings
      ↓
PostgreSQL + pgvector
```

### Recommendation flow

```text
User Preferences
      ↓
SentenceTransformer
      ↓
Query Embedding
      ↓
Vector Similarity Search
      ↓
PostgreSQL + pgvector
      ↓
Recommended Cocktails
```

Cosine similarity is used to find cocktails whose embeddings are closest to the user's query.

---

## Technologies

* Python
* Streamlit
* PostgreSQL
* pgvector
* Psycopg 3
* Sentence Transformers
* Hugging Face
* Pandas
* Docker
* Docker Compose

---


## Database Setup

After PostgreSQL is running, initialize the database and import the cocktail dataset:

```bash
python scripts/setup_database.py
```

The script performs the following steps:

```text
Create database
      ↓
Enable pgvector
      ↓
Create cocktails table
      ↓
Load CSV dataset
      ↓
Clean cocktail data
      ↓
Generate embeddings
      ↓
Store cocktails in PostgreSQL
```

This step usually only needs to be performed during the initial setup or when rebuilding the dataset.

---

## AI Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts text into 384-dimensional vector embeddings.

The first time the application initializes the model, the required model files are downloaded from Hugging Face and cached locally.

Future runs normally use the cached model.

---

## Possible Improvements

Future versions could include:

* User authentication
* Favorite cocktails
* Search history
* Cocktail images
* Advanced filtering
* Exact filtering combined with semantic search
* Real database statistics in the Streamlit dashboard
* Recommendation feedback
* Alternative embedding models
* ANN indexing for larger datasets
* REST API using FastAPI
* Deployment with Docker
* Automated testing and CI/CD

---


## Author

Developed as a learning project to explore:

* AI application architecture
* Semantic search
* Vector databases
* Sentence embeddings
* PostgreSQL and pgvector
* Python project organization
* Streamlit application development
