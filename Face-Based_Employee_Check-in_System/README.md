# Face-Based Employee Check-in System

A face-based employee check-in application built with **FaceNet**, **MTCNN**, **FAISS**, **OpenCV**, and **Streamlit**.

The project demonstrates how facial images can be converted into vector embeddings, stored and searched using a vector database, and used to identify employees for check-in.

It also includes a raw-pixel baseline for comparison and a PCA-based 3D visualization of the face embedding space.

---

## Features

* Load employee images from a local dataset
* Extract employee names automatically from image filenames
* Generate face embeddings using **FaceNet**
* Detect and crop faces using **MTCNN**
* Store and search face vectors using **FAISS**
* Compare two image retrieval approaches:

  * Raw Pixel Vector + L2 Distance
  * FaceNet Embedding + Cosine Similarity
* Recognize employees using similarity thresholds
* Capture images from the webcam using **OpenCV**
* Upload images manually as an alternative to camera capture
* Track employee check-in status using Streamlit Session State
* Display Top-K similar employees
* Visualize 512-dimensional face embeddings in 3D using **PCA + Plotly**
* Save and reload FAISS indexes without rebuilding them every time
* Support both:

  * Command-line application
  * Streamlit web application

---

## Project Structure

```text
Face-Based_Employee_Check-in_System/
│
├── app/
│   ├── main.py
│   └── streamlit_app.py
│
├── scripts/
│   └── build_indexes.py
│
├── src/
│   └── face_checkin/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── raw_pixels.py
│       ├── face_embeddings.py
│       ├── index_store.py
│       ├── visualization.py
│       └── system.py
│
├── data/
│   ├── raw/
│   │   └── Dataset/
│   ├── query/
│   └── processed/
│       └── indexes/
│
├── tests/
│   └── test_dataset.py
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## System Pipeline

### Approach 1: Raw Pixel Baseline

```text
Image
  ↓
Resize to 300 × 300
  ↓
Convert to NumPy Array
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

This approach compares images directly using pixel values.

It is included mainly as a baseline because raw pixels are highly sensitive to changes in lighting, pose, background, and image alignment.

---

### Approach 2: Face Embeddings

```text
Image
  ↓
MTCNN Face Detection
  ↓
Face Crop + Preprocessing
  ↓
InceptionResnetV1 / FaceNet
  ↓
512-Dimensional Embedding
  ↓
L2 Normalization
  ↓
FAISS IndexFlatIP
  ↓
Cosine Similarity Search
  ↓
Employee Recognition
```

This is the main recognition pipeline used by the application.

FaceNet represents each detected face as a **512-dimensional embedding**. The embeddings are L2-normalized before being stored in FAISS.

Because the vectors are normalized, the inner product returned by `IndexFlatIP` can be interpreted as cosine similarity.

Higher similarity means the faces are more similar.

---

## Vector Space Visualization

FaceNet produces vectors with 512 dimensions, which cannot be visualized directly.

The application therefore uses **Principal Component Analysis (PCA)** to reduce:

```text
512 dimensions
      ↓
3 principal components
      ↓
3D visualization
```

The reduced embeddings are displayed using Plotly.

The visualization contains:

* Registered employee embeddings
* The current query face
* The best match
* Other Top-K matches
* Lines connecting the query embedding to matching employees

PCA is used only for visualization. Employee recognition still uses the original 512-dimensional embeddings.

---

## Technologies

| Technology        | Purpose                                   |
| ----------------- | ----------------------------------------- |
| Python            | Main programming language                 |
| PyTorch           | Deep learning framework                   |
| facenet-pytorch   | MTCNN and FaceNet implementation          |
| MTCNN             | Face detection and cropping               |
| InceptionResnetV1 | Face embedding extraction                 |
| FAISS             | Vector indexing and similarity search     |
| NumPy             | Numerical operations                      |
| Pillow            | Image processing                          |
| OpenCV            | Webcam capture and video frame processing |
| Streamlit         | Web application interface                 |
| Scikit-learn      | PCA dimensionality reduction              |
| Plotly            | Interactive 3D visualization              |
| Pandas            | Tabular data preparation                  |
| Matplotlib        | Image/result visualization                |

---

## Requirements

The recommended environment for this project is:

```text
Python 3.11
```

The project currently uses versions compatible with `facenet-pytorch 2.6.0`.

Example dependencies:

```text
numpy==1.26.4
Pillow==10.2.0
faiss-cpu==1.8.0
torch==2.2.2
torchvision==0.17.2
facenet-pytorch==2.6.0

streamlit
opencv-python
pandas
plotly
scikit-learn
matplotlib
pytest
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Face-Based_Employee_Check-in_System
```

### 2. Create a virtual environment

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Upgrade installation tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install the project

```bash
pip install -e .
```

---

## Dataset Preparation

Place registered employee images inside:

```text
data/raw/Dataset/
```

Example:

```text
data/raw/Dataset/
├── Avatar_Alice.jpg
├── Avatar_Bob.jpg
├── Avatar_Charlie.jpg
└── Avatar_David.png
```

The employee label is extracted automatically from the filename.

For example:

```text
Avatar_Alice.jpg
```

becomes:

```text
Alice
```

Images without the `Avatar_` prefix are also supported.

---

## Build the FAISS Indexes

Before running the recognition application for the first time, build the vector indexes:

```bash
python scripts/build_indexes.py
```

The script creates two indexes:

```text
data/processed/indexes/
├── raw_pixels.index
├── raw_metadata.json
├── face_embeddings.index
└── face_metadata.json
```

The first index stores raw-pixel vectors, while the second stores FaceNet embeddings.

Saving these indexes means that the employee embeddings do not need to be regenerated every time the application starts.

---

## Run the CLI Application

Run:

```bash
python app/main.py
```

The CLI provides options such as:

```text
1. Build / rebuild indexes
2. Search using raw pixels
3. Search using face embeddings
4. Recognize employee using face embeddings
5. Recognize employee using raw pixels
0. Exit
```

---

## Run the Streamlit Application

Start the web application with:

```bash
python -m streamlit run app/streamlit_app.py
```

The Streamlit application provides:

* Employee list
* Check-in status
* Webcam capture
* Image upload
* Similarity threshold configuration
* Top-K face matches
* Check-in result
* PCA-based vector space visualization
* Similarity table

---

## Webcam Check-in

The webcam feature uses OpenCV:

```python
cv2.VideoCapture(0)
```

Frames are captured from the camera using:

```python
ret, frame = cap.read()
```

OpenCV stores images in **BGR** channel order, while Streamlit and Pillow normally use **RGB**.

Therefore, captured frames are converted using:

```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

The final captured frame is then sent through the same MTCNN → FaceNet → FAISS pipeline as an uploaded image.

---

## Similarity Search

Face embeddings are normalized before being added to the FAISS index.

The face index uses:

```python
faiss.IndexFlatIP(512)
```

For normalized vectors:

```text
Inner Product ≈ Cosine Similarity
```

Therefore:

```text
Higher similarity → more similar faces
Lower similarity  → less similar faces
```

The application uses a configurable similarity threshold to determine whether the best match should be accepted as a recognized employee.

The threshold should ideally be calibrated using validation images rather than treated as a universal constant.

---

## Raw Pixels vs Face Embeddings

| Approach   | Vector Size | FAISS Metric                      | Main Advantage                              | Main Limitation                             |
| ---------- | ----------: | --------------------------------- | ------------------------------------------- | ------------------------------------------- |
| Raw Pixels |     270,000 | L2 Distance                       | Simple baseline                             | Sensitive to pose, lighting, and background |
| FaceNet    |         512 | Inner Product / Cosine Similarity | Represents facial identity more effectively | Requires a pretrained deep-learning model   |

The raw-pixel approach helps demonstrate why feature representation is important in similarity-search systems.

---

## macOS Notes

On Apple Silicon Macs, PyTorch may detect the MPS backend.

However, some operations used by MTCNN can produce errors such as:

```text
RuntimeError: Adaptive pool MPS:
input sizes must be divisible by output sizes
```

For this project, CPU execution is used on macOS for better compatibility with `facenet-pytorch`.

Example device configuration:

```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
```

This means:

```text
NVIDIA GPU → CUDA
Mac / Other → CPU
```

For a small employee dataset, CPU inference is sufficient for this project.

---

## Architecture

The project separates application code from core recognition logic.

```text
Streamlit / CLI
       ↓
FaceCheckinSystem
       │
       ├── Dataset Loader
       │
       ├── Raw Pixel Pipeline
       │       └── FAISS L2
       │
       ├── FaceEmbeddingEngine
       │       ├── MTCNN
       │       ├── FaceNet
       │       └── FAISS Inner Product
       │
       └── Index Storage
```

This structure prevents the UI layer from containing model-loading, preprocessing, indexing, and recognition logic.

Both the CLI and Streamlit application reuse the same underlying recognition system.

---

## Main Modules

### `dataset.py`

Loads employee images and creates image-label records.

### `raw_pixels.py`

Handles:

* Image resizing
* Pixel normalization
* Vector flattening
* Raw-pixel FAISS indexing
* L2 similarity search

### `face_embeddings.py`

Handles:

* MTCNN face detection
* FaceNet embedding extraction
* L2 normalization
* Face embedding indexing
* Similarity search
* Recognition

### `index_store.py`

Handles saving and loading:

* FAISS indexes
* Associated metadata

### `system.py`

Provides the main interface between the application layer and the face recognition components.

### `streamlit_app.py`

Provides:

* Web UI
* Camera interaction
* Session state
* Check-in workflow
* Top matches
* PCA visualization

---

## Testing

Run the test suite with:

```bash
pytest
```

The current tests include dataset-loading behavior such as:

* Supported image extensions
* Automatic employee label extraction
* Ignoring unsupported files

---

## Current Limitations

* Employee images are currently loaded from a local dataset
* Check-in status is stored only in Streamlit Session State
* Check-in history is not persisted to a database
* Recognition threshold still needs dataset-specific calibration
* The application assumes one main face per query image
* CPU inference may become slow for larger workloads
* The current webcam implementation captures a single image rather than performing continuous real-time recognition

---

## Possible Future Improvements

Future versions could include:

* Real-time face recognition from continuous webcam frames
* Face bounding boxes and employee names drawn directly on video
* Persistent employee and attendance database
* Check-in timestamps and attendance history
* Automatic similarity-threshold calibration
* Multiple reference images per employee
* Unknown-person detection
* REST API for the recognition service
* Improved evaluation using verification metrics
* Docker deployment
* Streamlit Cloud or server deployment

---

## Learning Objectives

This project was developed as a practical exercise in:

* Vector databases
* Image embeddings
* Face recognition
* Similarity search
* FAISS indexing
* Transfer learning with pretrained models
* Computer vision preprocessing
* PCA dimensionality reduction
* OpenCV webcam processing
* Streamlit application development
* Structuring a machine-learning project into reusable modules

---

## License

This project is intended for educational and learning purposes.

Before using face recognition in a real-world environment, privacy, consent, security, and applicable biometric-data regulations should be considered.
