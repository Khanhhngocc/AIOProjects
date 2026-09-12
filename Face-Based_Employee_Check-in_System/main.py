# ============================================================
# FACE-BASED EMPLOYEE CHECK-IN SYSTEM
# Chapter 9 - Vector Database
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import faiss

import torch
from torchvision import transforms

from facenet_pytorch import InceptionResnetV1

from tqdm import tqdm


# ============================================================
# 2. DATASET PREPARATION
# ============================================================
dataset_path = "Dataset"

image_paths = []
labels = []


# ------------------------------------------------------------
# Extract image paths and employee labels
# ------------------------------------------------------------

for filename in os.listdir(dataset_path):

    if filename.endswith((".jpg", ".JPG", ".png", ".jpeg")):

        # Store full image path
        image_paths.append(
            os.path.join(dataset_path, filename)
        )

        # Remove file extension
        file_name = filename.split(".")[0]

        # Remove "Avatar_" prefix
        label = file_name[7:]

        labels.append(label)


# ------------------------------------------------------------
# Create DataFrame
#
# The lecture later uses df.iterrows(),
# so we combine image_paths and labels into a DataFrame here.
# ------------------------------------------------------------

df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})

print(df.head())
print(f"Number of employee images: {len(df)}")


# ============================================================
# PART I
# RAW PIXEL VECTOR
# ============================================================


# ============================================================
# 3. RAW PIXEL PREPROCESSING CONFIGURATION
# ============================================================

IMAGE_SIZE = 300
VECTOR_DIM = 300 * 300 * 3


# ============================================================
# 4. CREATE FAISS INDEX FOR RAW PIXELS
# ============================================================

# IndexFlatL2 uses L2 / Euclidean distance.
index = faiss.IndexFlatL2(VECTOR_DIM)

# label_map[i] corresponds to vector i inside FAISS.
label_map = []


# ============================================================
# 5. CONVERT IMAGE TO RAW PIXEL VECTOR
# ============================================================

def image_to_vector(image_path):

    # Load image and resize to 300x300
    img = Image.open(image_path).resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert image to NumPy array
    img_array = np.array(img)

    # --------------------------------------------------------
    # Handle grayscale image
    # --------------------------------------------------------

    # Grayscale image:
    # shape = (H, W)
    #
    # RGB image:
    # shape = (H, W, 3)

    if len(img_array.shape) == 2:

        img_array = np.stack(
            (img_array,) * 3,
            axis=-1
        )


    # --------------------------------------------------------
    # Normalize pixel values
    # --------------------------------------------------------

    # Original:
    # pixel values = 0 -> 255
    #
    # After normalization:
    # pixel values = 0 -> 1

    vector = img_array.astype("float32") / 255.0


    # --------------------------------------------------------
    # Flatten image
    # --------------------------------------------------------

    # Before:
    # (300, 300, 3)
    #
    # After:
    # (270000,)
    return vector.flatten()


# ============================================================
# 6. BUILD RAW PIXEL FAISS INDEX
# ============================================================

for idx, row in df.iterrows():
    image_path = row["image_path"]
    label = row["label"]

    try:

        # Convert image -> raw pixel vector
        vector = image_to_vector(image_path)

        # FAISS expects:
        # (number_of_vectors, vector_dimension)
        #
        # vector:
        # (270000,)
        #
        # np.array([vector]):
        # (1, 270000)

        index.add(np.array([vector]))

        # Save employee name corresponding to this vector
        label_map.append(label)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# ============================================================
# 7. SAVE RAW PIXEL FAISS INDEX
# ============================================================

faiss.write_index(
    index,
    "employee_images.index"
)

np.save(
    "label_map.npy",
    np.array(label_map)
)


# ============================================================
# 8. SEARCH SIMILAR IMAGES USING RAW PIXELS
# ============================================================

def search_similar_images_raw(query_image_path, k=5):

    """
    Search for similar employee images
    using raw pixel vectors.
    """

    # --------------------------------------------------------
    # Load FAISS index
    # --------------------------------------------------------

    index = faiss.read_index(
        "employee_images.index"
    )

    # Load employee labels
    label_map = np.load(
        "label_map.npy"
    )


    # --------------------------------------------------------
    # Convert query image to vector
    # --------------------------------------------------------

    query_vector = image_to_vector(
        query_image_path
    )


    # --------------------------------------------------------
    # Search nearest vectors in FAISS
    # --------------------------------------------------------

    distances, indices = index.search(
        np.array([query_vector]),
        k
    )


    # --------------------------------------------------------
    # Convert FAISS results to employee names
    # --------------------------------------------------------

    results = []

    for i in range(len(indices[0])):

        employee_name = label_map[
            indices[0][i]
        ]

        distance = distances[0][i]

        results.append(
            (
                employee_name,
                distance
            )
        )

    return results


# ============================================================
# 9. DISPLAY RAW PIXEL SEARCH RESULTS
# ============================================================

def display_raw_pixel_matches(query_image_path):

    # --------------------------------------------------------
    # Display query image
    # --------------------------------------------------------

    query_img = Image.open(
        query_image_path
    )

    query_img = query_img.resize(
        (300, 300)
    )

    plt.figure(figsize=(5, 5))

    plt.imshow(query_img)

    plt.title(
        "Query Image"
    )

    plt.axis("off")

    plt.show()


    # --------------------------------------------------------
    # Search top 5 nearest employee images
    # --------------------------------------------------------

    matches = search_similar_images_raw(
        query_image_path
    )


    # --------------------------------------------------------
    # Find image paths corresponding to employee names
    # --------------------------------------------------------

    top_matches = []

    for name, distance in matches:

        img_path = df[
            df["label"] == name
        ]["image_path"].values[0]

        top_matches.append(
            (
                name,
                distance,
                img_path
            )
        )


    # --------------------------------------------------------
    # Display top matches
    # --------------------------------------------------------

    plt.figure(
        figsize=(15, 5)
    )

    for i, (name, distance, img_path) in enumerate(top_matches):

        img = Image.open(img_path)

        img = img.resize((300, 300))

        plt.subplot(1, 5, i + 1)

        plt.imshow(img)

        plt.title(
            f"{name}\n"
            f"Dist: {distance:.2f}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# PART II
# FACENET FEATURE EMBEDDING
# ============================================================


# ============================================================
# 10. LOAD PRETRAINED FACE RECOGNITION MODEL
# ============================================================

# InceptionResnetV1 pretrained on VGGFace2.
#
# We do NOT train the model again.
# It is used only as a feature extractor.

face_recognition_model = InceptionResnetV1(
    pretrained="vggface2"
).eval()


# ============================================================
# 11. IMAGE PREPROCESSING FOR FACENET
# ============================================================

transform = transforms.Compose([

    # Resize image
    transforms.Resize((300, 300)),

    # PIL Image -> PyTorch Tensor
    transforms.ToTensor(),

    # Normalize tensor
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

])


# ============================================================
# 12. EXTRACT FACE FEATURE VECTOR
# ============================================================

def extract_feature(image_path, model):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    img = Image.open(image_path).convert("RGB")


    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    img_tensor = transform(img)


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    # Before:
    # (C, H, W)
    #
    # After:
    # (1, C, H, W)

    img_tensor = img_tensor.unsqueeze(0)


    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    # We are doing inference only,
    # so gradients are not required.

    with torch.no_grad():

        features = model(img_tensor)


    # --------------------------------------------------------
    # PyTorch Tensor -> NumPy vector
    # --------------------------------------------------------

    return features.squeeze().numpy()


# ============================================================
# 13. CREATE FAISS INDEX FOR FACENET FEATURES
# ============================================================

# FaceNet produces a 512-dimensional feature vector.
VECTOR_DIM = 512


# IndexFlatIP:
# Inner Product similarity
feature_index = faiss.IndexFlatIP(
    VECTOR_DIM
)

feature_label_map = []


# ============================================================
# 14. BUILD FACENET FEATURE INDEX
# ============================================================

for idx, row in tqdm(
    df.iterrows(),
    total=len(df)
):

    try:

        # ----------------------------------------------------
        # Image -> FaceNet embedding
        # ----------------------------------------------------

        features = extract_feature(
            row["image_path"],
            face_recognition_model
        )


        # ----------------------------------------------------
        # Add embedding to FAISS
        # ----------------------------------------------------

        feature_index.add(
            np.array([features])
        )


        # ----------------------------------------------------
        # Save employee label
        # ----------------------------------------------------

        feature_label_map.append(
            row["label"]
        )

    except Exception as e:
        print(
            f"Error processing "
            f"{row['image_path']}: {e}"
        )


# ============================================================
# 15. SAVE FACENET FAISS INDEX
# ============================================================

faiss.write_index(
    feature_index,
    "facenet_features.index"
)

np.save(
    "facenet_label_map.npy",
    np.array(feature_label_map)
)


# ============================================================
# 16. CONVERT QUERY IMAGE TO FACENET FEATURE
# ============================================================

def image_to_feature(image_path, model):

    """
    Convert image to face embedding
    using a pretrained model.
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    img = Image.open(image_path).convert("RGB")


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    img_tensor = transform(img).unsqueeze(0)


    # --------------------------------------------------------
    # Extract embedding
    # --------------------------------------------------------

    with torch.no_grad():
        embedding = model(img_tensor)


    # --------------------------------------------------------
    # Tensor -> NumPy
    # --------------------------------------------------------

    return embedding.squeeze().numpy()


# ============================================================
# 17. SEARCH USING FACENET FEATURES
# ============================================================

def search_similar_images_feature(query_image_path, k=5):

    """
    Search for similar employee images
    using FaceNet features.
    """

    # --------------------------------------------------------
    # Load FAISS index and labels
    # --------------------------------------------------------

    index = faiss.read_index(
        "facenet_features.index"
    )

    label_map = np.load(
        "facenet_label_map.npy"
    )


    # --------------------------------------------------------
    # Query image -> FaceNet embedding
    # --------------------------------------------------------

    query_vector = image_to_feature(
        query_image_path,
        face_recognition_model
    )


    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    similarities, indices = index.search(
        np.array([query_vector]),
        k
    )


    # --------------------------------------------------------
    # Build result list
    # --------------------------------------------------------

    results = []

    for i in range(
        len(indices[0])
    ):

        employee_name = label_map[
            indices[0][i]
        ]

        similarity = similarities[0][i]

        results.append(
            (
                employee_name,
                similarity
            )
        )

    return results


# ============================================================
# 18. DISPLAY FACENET SEARCH RESULTS
# ============================================================

def display_feature_matches(query_image_path):

    # --------------------------------------------------------
    # Display query image
    # --------------------------------------------------------

    query_img = Image.open(
        query_image_path
    )

    query_img = query_img.resize(
        (300, 300)
    )

    plt.figure(figsize=(5, 5))

    plt.imshow(query_img)

    plt.title("Query Image")

    plt.axis("off")

    plt.show()


    # --------------------------------------------------------
    # Get top matches
    # --------------------------------------------------------

    matches = search_similar_images_feature(
        query_image_path
    )


    # --------------------------------------------------------
    # Display matching employees
    # --------------------------------------------------------

    plt.figure(figsize=(15, 5))

    for i, (name, similarity) in enumerate(matches):


        # Find employee image path
        img_path = df[
            df["label"] == name
        ]["image_path"].values[0]


        img = Image.open(img_path)

        img = img.resize((300, 300))


        plt.subplot(1, 5, i + 1)

        plt.imshow(img)

        plt.title(
            f"{name}\n"
            f"Similarity: {similarity:.2f}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# 19. TEST
# ============================================================

# Example:
#
# query_image_path = "test_image.jpg"


# ------------------------------------------------------------
# Test Level 1 - Raw Pixel Vector
# ------------------------------------------------------------

# display_raw_pixel_matches(
#     query_image_path
# )


# ------------------------------------------------------------
# Test Level 2 - FaceNet Feature Vector
# ------------------------------------------------------------

# display_feature_matches(
#     query_image_path
# )