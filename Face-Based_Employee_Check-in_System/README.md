# Face-Based Employee Check-in System

A face-based employee check-in application built with **FaceNet**, **MTCNN**, **FAISS**, **OpenCV**, and **Streamlit**.

The project demonstrates how facial images can be converted into vector embeddings, stored and searched using a vector database, and used to identify employees for check-in.

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
| Pandas            | Tabular data preparation                  |
| Matplotlib        | Image/result visualization                |

---

## Raw Pixels vs Face Embeddings

| Approach   | Vector Size | FAISS Metric                      | Main Advantage                              | Main Limitation                             |
| ---------- | ----------: | --------------------------------- | ------------------------------------------- | ------------------------------------------- |
| Raw Pixels |     270,000 | L2 Distance                       | Simple baseline                             | Sensitive to pose, lighting, and background |
| FaceNet    |         512 | Inner Product / Cosine Similarity | Represents facial identity more effectively | Requires a pretrained deep-learning model   |

The raw-pixel approach helps demonstrate why feature representation is important in similarity-search systems.

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