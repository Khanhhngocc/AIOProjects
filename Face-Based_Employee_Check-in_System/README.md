# Face-Based Employee Check-in System

A face-based employee check-in application built with **InsightFace**, **FAISS**, **OpenCV**, and **Streamlit**.

The project demonstrates how facial images can be converted into vector embeddings, searched using FAISS, and used to identify employees for check-in.

The project was initially developed with **MTCNN + FaceNet** as a learning implementation, then migrated to **InsightFace + ONNX Runtime** for better compatibility and deployment.

---

## Features

- Load employee images from a local dataset
- Extract employee names automatically from image filenames
- Detect faces and generate **512-dimensional face embeddings** using InsightFace
- Store and search face vectors using **FAISS**
- Compare two image retrieval approaches:
  - Raw Pixel Vector + L2 Distance
  - Face Embedding + Cosine Similarity
- Recognize employees using a similarity threshold
- Capture images using the browser camera
- Upload images manually for recognition
- Track employee check-in status using Streamlit Session State
- Display Top-K similar employees
- Visualize 512-dimensional face embeddings in 3D using **PCA + Plotly**
- Automatically build the FAISS face index when a saved index is unavailable
- Support both command-line and Streamlit applications

---

## System Pipeline

### Approach 1: Raw Pixel Baseline

```text
Image
  ↓
Resize to 300 × 300
  ↓
Normalize Pixel Values
  ↓
Flatten
  ↓
270,000-Dimensional Vector
  ↓
FAISS IndexFlatL2
  ↓
Nearest-Neighbor Search
```

This approach is included as a baseline. Raw pixel vectors are highly sensitive to lighting, pose, background, and image alignment.

---

### Approach 2: Face Embeddings

```text
Image
  ↓
InsightFace
  ↓
Face Detection + Recognition
  ↓
512-Dimensional Normalized Embedding
  ↓
FAISS IndexFlatIP
  ↓
Cosine Similarity Search
  ↓
Employee Recognition
```

This is the main recognition pipeline used by the application.

InsightFace converts each detected face into a **512-dimensional normalized embedding**. Because the vectors are normalized, the inner product returned by `IndexFlatIP` can be used as cosine similarity.

Higher similarity indicates more similar facial embeddings.

---

## From FaceNet to InsightFace

The first version of this project used:

```text
MTCNN
→ InceptionResnetV1 / FaceNet
→ 512-D Embedding
→ FAISS
```

This version was useful for learning the individual steps of face detection, preprocessing, embedding extraction, and similarity search.

The project was later migrated to:

```text
InsightFace
→ Face Detection + Face Recognition
→ 512-D Embedding
→ FAISS
```

The migration kept the main vector-search architecture unchanged while replacing the embedding model and inference backend.

This helped demonstrate an important concept: **the embedding model can change while the FAISS similarity-search pipeline remains reusable.**

---

## Technologies

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| InsightFace | Face detection and face embedding extraction |
| ONNX Runtime | Model inference backend |
| FAISS | Vector indexing and similarity search |
| OpenCV | Image decoding and preprocessing |
| Streamlit | Web application interface |
| NumPy | Numerical operations |
| Scikit-learn | PCA dimensionality reduction |
| Plotly | Interactive 3D embedding visualization |
| Pandas | Tabular data preparation |
| Pillow | Image processing |

---

## Raw Pixels vs Face Embeddings

| Approach | Vector Size | Similarity Method | Main Limitation |
| --- | ---: | --- | --- |
| Raw Pixels | 270,000 | L2 Distance | Sensitive to lighting, pose, and background |
| Face Embeddings | 512 | Cosine Similarity | Requires a pretrained face-recognition model |

The comparison demonstrates why learned feature representations are more suitable than raw pixels for facial similarity search.

---

## Current Limitations

- Employee images are stored in a local project dataset
- Check-in status is stored only in Streamlit Session State
- Check-in history is not persisted to a database
- The recognition threshold still requires dataset-specific calibration
- The application assumes one main face per query image
- The camera captures a single image instead of continuous real-time recognition

---

## Possible Future Improvements

- Persistent employee and attendance database
- Check-in timestamps and attendance history
- Multiple reference images per employee
- Automatic similarity-threshold calibration
- Unknown-person detection
- Real-time face recognition from continuous video
- REST API for the recognition service

---

## Learning Objectives

This project was developed as a practical exercise in:

- Face recognition and image embeddings
- Vector similarity search
- FAISS indexing
- Computer vision preprocessing
- PCA dimensionality reduction
- Streamlit application development
- Deploying an AI application
- Migrating from one pretrained face-recognition stack to another
- Structuring a machine-learning project into reusable modules

---

## License

This project is intended for educational and learning purposes.