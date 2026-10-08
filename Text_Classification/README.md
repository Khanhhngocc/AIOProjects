# Spam Classifier with E5 Embeddings, FAISS, and KNN

A text classification project that detects whether a message is **spam** or **ham** using semantic embeddings, vector similarity search, and K-Nearest Neighbors (KNN).

Instead of training a traditional classification model, this project converts messages into dense vector representations using the multilingual E5 embedding model, stores training embeddings in a FAISS vector index, and classifies new messages based on their nearest neighbors.

## Overview

The classification pipeline follows this workflow:

```text
Input Message
     ↓
E5 Text Embedding
     ↓
FAISS Similarity Search
     ↓
Top-K Nearest Neighbors
     ↓
Majority Voting
     ↓
Spam / Ham Prediction
```

The project also includes an evaluation pipeline for comparing different values of `k` and saving misclassified examples for error analysis.

## Features

* Text embedding with `intfloat/multilingual-e5-base`
* Batched embedding generation with PyTorch
* Automatic device selection:

  * CUDA
  * Apple Silicon MPS
  * CPU
* L2-normalized sentence embeddings
* FAISS vector similarity search
* KNN-based text classification
* Majority-vote prediction
* Evaluation with multiple `k` values
* Nearest-neighbor inspection for each prediction
* Error analysis exported to JSON
* Command-line interface for interactive testing
* Modular project structure
* Basic unit tests with Pytest

## Project Structure

```text
spam_classifier_project/
├── app/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── 2cls_spam_text_cls.csv
│
├── outputs/
│   └── error_analysis.json
│
├── src/
│   └── spam_classifier/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── embedder.py
│       ├── vector_store.py
│       ├── classifier.py
│       ├── evaluator.py
│       └── pipeline.py
│
├── tests/
│   └── test_classifier.py
│
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Main Components

### E5 Embedder

The project uses:

```text
intfloat/multilingual-e5-base
```

to convert messages into dense semantic vectors.

E5 uses different prefixes depending on the role of the text:

```text
passage: message stored in the training database
query: new message used for searching
```

Token-level representations from the Transformer are converted into sentence embeddings using average pooling.

The resulting embeddings are then L2-normalized.

### FAISS Vector Store

Training embeddings are stored in a FAISS index using:

```python
faiss.IndexFlatIP
```

Because the embeddings are L2-normalized, inner-product similarity can be used as an approximation of cosine similarity.

When a new message is received, FAISS retrieves the `k` most similar messages from the training set.

### KNN Classification

The labels of the nearest neighbors are collected and the final prediction is determined using majority voting.

For example:

```text
Neighbor 1 → spam
Neighbor 2 → spam
Neighbor 3 → ham

Final prediction → spam
```

The classifier also returns similarity scores and neighbor messages so that predictions can be inspected.

## Dataset

Place the dataset inside:

```text
data/2cls_spam_text_cls.csv
```

The CSV file must contain the following columns:

| Column     | Description                           |
| ---------- | ------------------------------------- |
| `Message`  | Message text                          |
| `Category` | Message label such as `ham` or `spam` |

Example:

```csv
Category,Message
ham,Are we still meeting this afternoon?
spam,Congratulations! You have won a free prize!
```

The dataset is split into training and test sets using stratified sampling to preserve the label distribution.

Default configuration:

```python
TEST_SIZE = 0.1
RANDOM_STATE = 42
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd spam_classifier_project
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the project in editable mode

```bash
pip install -e .
```

## Requirements

Main dependencies include:

```text
numpy
pandas
torch
transformers
faiss-cpu
scikit-learn
tqdm
pytest
```

## Usage

Run the application from the project root:

```bash
python3 -m app.main
```

The application will first:

```text
1. Load the dataset
2. Load the E5 embedding model
3. Generate message embeddings
4. Split the dataset
5. Build the FAISS index
6. Initialize the KNN classifier
```

After initialization, the CLI displays:

```text
============================================================
SPAM CLASSIFIER
============================================================

1. Classify a message
2. Evaluate classifier
3. Exit
```

### Classify a Message

Choose:

```text
1. Classify a message
```

Then enter a message:

```text
Enter message: Win a free iPhone! Click here now!
Enter k [default=3]: 3
```

Example output:

```text
============================================================
Prediction: SPAM
Label distribution: {'spam': 3}

Top neighbors:

1. Label: spam | Score: 0.9214
   Message: Congratulations! You have won...

2. Label: spam | Score: 0.8992
   Message: Free prize waiting for you...

3. Label: spam | Score: 0.8745
   Message: Claim your reward now...
============================================================
```

The nearest-neighbor output makes the prediction easier to inspect because users can see which training messages influenced the classification result.

## Evaluation

Choose:

```text
2. Evaluate classifier
```

The current evaluation compares:

```text
k = 1
k = 3
k = 5
```

Example:

```text
k=1 | Accuracy: 0.9643 (96.43%) | Errors: 20/560
k=3 | Accuracy: 0.9714 (97.14%) | Errors: 16/560
k=5 | Accuracy: 0.9679 (96.79%) | Errors: 18/560
```

Actual results depend on the dataset and environment.

## Error Analysis

Misclassified messages are saved automatically to:

```text
outputs/error_analysis.json
```

The file contains:

* Evaluation timestamp
* Embedding model
* Test-set size
* Accuracy for each value of `k`
* Number of errors
* Error rate
* True label
* Predicted label
* Nearest neighbors
* Similarity scores
* Neighbor label distribution

This makes it easier to investigate why particular messages were classified incorrectly.

## Running Tests

Run the test suite with:

```bash
pytest
```

The current tests focus on core classification logic such as majority voting.

## Configuration

Main project settings are located in:

```text
src/spam_classifier/config.py
```

Examples include:

```python
MODEL_NAME = "intfloat/multilingual-e5-base"

TEST_SIZE = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 32
DEFAULT_K = 3
```

These values can be changed without modifying the main classification pipeline.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* E5 Embeddings
* FAISS
* NumPy
* Pandas
* Scikit-learn
* Pytest

## Key Concepts Demonstrated

This project demonstrates several concepts commonly used in modern NLP and retrieval systems:

* Transformer-based text embeddings
* Semantic similarity
* Vector databases
* Cosine similarity
* K-Nearest Neighbors
* Majority voting
* Batched inference
* Train/test splitting
* Model evaluation
* Error analysis
* Modular Python project design

Unlike a conventional neural-network classifier, this approach performs classification through semantic retrieval. This also makes individual predictions more interpretable because the nearest training examples can be inspected directly.

## Limitations

The current implementation has several limitations:

* Embeddings are regenerated whenever the application starts.
* The FAISS index is rebuilt on every run.
* Classification quality depends strongly on the training examples.
* Majority voting treats all neighbors equally regardless of their similarity scores.
* There is currently no confidence threshold for uncertain predictions.
* The current interface is command-line based.

## Possible Improvements

Future versions could include:

* Saving and loading precomputed embeddings
* Persisting the FAISS index
* Similarity-weighted KNN voting
* Automatic selection of the best `k`
* Precision, recall, and F1-score evaluation
* Confusion matrix visualization
* Confidence thresholds
* Streamlit web interface
* REST API deployment
* Comparison with traditional machine-learning classifiers
* Comparison with fine-tuned Transformer classifiers

## Learning Purpose

This project was developed as a practical exercise in building an end-to-end NLP pipeline using modern text embeddings and vector similarity search.

The main goal is to understand how raw text can be transformed into semantic vector representations and used for retrieval-based classification without training a dedicated neural classification model.

## License

This project is intended for educational and portfolio purposes.
