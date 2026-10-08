# 🍹 Cocktail Suggestion System

An AI-powered cocktail recommendation system built with **Python**, **Sentence Transformers**, **PostgreSQL**, **pgvector**, and **Streamlit**.

The application converts cocktail information and user preferences into vector embeddings and uses semantic similarity search to recommend cocktails based on ingredients, style, occasion, category, or mixed preferences.

---

## ✨ Features

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

## 🧠 How It Works

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

## 🏗 Project Structure

```text
Cocktail_Suggestions/
│
├── app/
│   ├── main.py
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── cocktails.csv
│   └── processed/
│
├── notebooks/
│
├── scripts/
│   └── setup_database.py
│
├── src/
│   └── cocktail_suggestions/
│       ├── __init__.py
│       ├── config.py
│       ├── database.py
│       ├── data_processor.py
│       └── recommender.py
│
├── tests/
│
├── .env
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Main directories

* `app/` — Application entry points for CLI and Streamlit
* `src/` — Core application logic
* `data/` — Raw and processed datasets
* `scripts/` — Utility scripts such as database initialization
* `notebooks/` — Experiments, data exploration, and embedding analysis
* `tests/` — Unit and integration tests

---

## 🧩 Main Components

### `config.py`

Loads configuration values from environment variables.

Examples:

* PostgreSQL host
* PostgreSQL port
* Database name
* Database credentials
* Sentence Transformer model name

---

### `database.py`

Handles PostgreSQL and pgvector operations.

Main responsibilities:

* Connect to PostgreSQL
* Create the project database
* Enable the pgvector extension
* Create the `cocktails` table

---

### `data_processor.py`

Handles dataset preprocessing and embedding generation.

Main responsibilities:

* Clean raw cocktail data
* Normalize supported dataset schemas
* Create combined semantic text
* Generate embeddings
* Build readable cocktail recipes
* Store cocktails and embeddings in PostgreSQL

---

### `recommender.py`

Contains the recommendation and search logic.

Supported operations include:

* Search by name
* Search by ingredients
* Search by style
* Search by occasion
* Mixed-preference recommendation
* Category browsing
* Random cocktail discovery
* Vector similarity search

---

### `streamlit_app.py`

Provides the web interface using Streamlit.

The UI allows users to select different search modes and interact with the recommendation system through a browser.

---

## 🛠 Technologies

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

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Cocktail_Suggestions
```

---

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

---

### 3. Install the project

```bash
pip install -e .
```

The editable installation allows Python to import the package directly from:

```text
src/cocktail_suggestions/
```

Changes to the source code are therefore available immediately without reinstalling the project.

If necessary, dependencies can also be installed from:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root.

Example:

```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=cocktail_db

MODEL_NAME=all-MiniLM-L6-v2
```

Do not commit the real `.env` file to Git.

A `.env.example` file can be included instead:

```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=
DB_PASSWORD=
DB_NAME=cocktail_db

MODEL_NAME=all-MiniLM-L6-v2
```

---

## 🐳 PostgreSQL with Docker

The project uses PostgreSQL with the pgvector extension.

Start the database container:

```bash
docker compose up -d
```

Check its status:

```bash
docker compose ps
```

The PostgreSQL server should become available at:

```text
localhost:5432
```

To stop the container:

```bash
docker compose stop
```

To start it again:

```bash
docker compose start
```

To remove the container while keeping database data:

```bash
docker compose down
```

> Avoid using `docker compose down -v` unless you intentionally want to remove the PostgreSQL volume and database data.

---

## 🗄 Database Setup

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

## 🤖 AI Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts text into 384-dimensional vector embeddings.

The first time the application initializes the model, the required model files are downloaded from Hugging Face and cached locally.

Future runs normally use the cached model.

---

## 🌐 Running the Streamlit App

Make sure PostgreSQL is running:

```bash
docker compose up -d
```

Then start Streamlit:

```bash
streamlit run app/streamlit_app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

The Streamlit interface is the main user-facing application.

---

## 💻 Running the CLI Version

A simple terminal interface is also available:

```bash
python app/main.py
```

The CLI and Streamlit applications both use the same recommendation logic from:

```text
src/cocktail_suggestions/
```

Therefore, running both interfaces at the same time is not required.

---

## 🔄 Typical Development Workflow

After the first setup, the usual workflow is:

```bash
docker compose up -d
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Then run the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

There is normally no need to run:

```bash
pip install -e .
```

again unless the development environment is recreated or project dependencies change.

There is also no need to rerun:

```bash
python scripts/setup_database.py
```

unless the database needs to be rebuilt or the source dataset changes.

---

## 🔍 Example Recommendation Flow

Suppose the user selects:

```text
lime
mint
refreshing
summer evening
```

The application builds a semantic query such as:

```text
contains lime and mint is refreshing perfect for summer evening
```

The query is transformed into an embedding:

```text
User preferences
      ↓
all-MiniLM-L6-v2
      ↓
384-dimensional vector
```

PostgreSQL then compares the query vector with the stored cocktail vectors using pgvector cosine distance.

The cocktails with the highest semantic similarity are returned first.

---

## 🧪 Testing

Tests can be placed inside:

```text
tests/
```

Run them with:

```bash
pytest
```

Possible tests include:

* Data cleaning
* Duplicate removal
* Recipe generation
* Search result formatting
* Database operations
* Recommendation functions

---

## 📓 Notebooks

The `notebooks/` directory can be used for experimentation and learning.

Examples:

```text
notebooks/
├── 01_data_exploration.ipynb
└── 02_embedding_experiment.ipynb
```

Recommended uses include:

* Dataset exploration
* Missing-value analysis
* Category distribution
* Embedding inspection
* Semantic similarity experiments
* Model comparison

Experimental notebook code should be moved into `src/` once it becomes part of the actual application.

---

## 🧱 Architecture

```text
                         User
                          │
                          ▼
                Streamlit / CLI
                          │
                          ▼
                CocktailRecommender
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
     SentenceTransformer        DatabaseSetup
              │                       │
              ▼                       ▼
     all-MiniLM-L6-v2              Psycopg
                                      │
                                      ▼
                             PostgreSQL + pgvector
                                      │
                                      ▼
                              Vector Similarity
                                      │
                                      ▼
                               Cocktail Results
```

---

## 📌 Important Notes

* PostgreSQL must be running before using the application.
* The pgvector extension must exist in the project database.
* Run `setup_database.py` before the first application run.
* The pretrained model is downloaded automatically the first time it is used.
* `.env`, virtual environments, caches, and generated package metadata should not be committed to Git.

Recommended `.gitignore` entries:

```gitignore
.venv/
.env
__pycache__/
*.pyc
*.egg-info/
.ipynb_checkpoints/
.DS_Store
.idea/
.vscode/
```

---

## 🔮 Possible Improvements

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

## 📄 License

This project is intended for educational and portfolio purposes.

---

## 👤 Author

Developed as a learning project to explore:

* AI application architecture
* Semantic search
* Vector databases
* Sentence embeddings
* PostgreSQL and pgvector
* Python project organization
* Streamlit application development
