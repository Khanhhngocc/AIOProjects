# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import faiss

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# ============================================================
# 2. READ DATASET
# ============================================================

DATASET_PATH = "/content/2cls_spam_text_cls.csv"

df = pd.read_csv(DATASET_PATH)

messages = df["Message"].values.tolist()
labels = df["Category"].values.tolist()


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


# ============================================================
# 4. SET DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)
model.eval()


# ============================================================
# 5. AVERAGE POOLING
# ============================================================

def average_pool(
    last_hidden_states,
    attention_mask
):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(),
        0.0
    )

    return (
        last_hidden.sum(dim=1)
        / attention_mask.sum(dim=1)[..., None]
    )


# ============================================================
# 6. CREATE SENTENCE EMBEDDINGS
# ============================================================

def get_embeddings(
    texts,
    model,
    tokenizer,
    device,
    batch_size=32
):
    """
    Create embeddings for a list of texts.
    """

    embeddings = []

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Generating embeddings"
    ):
        batch_texts = texts[i:i + batch_size]

        batch_texts_with_prefix = [
            f"passage: {text}"
            for text in batch_texts
        ]

        batch_dict = tokenizer(
            batch_texts_with_prefix,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        batch_dict = {
            k: v.to(device)
            for k, v in batch_dict.items()
        }

        with torch.no_grad():
            outputs = model(**batch_dict)

            batch_embeddings = average_pool(
                outputs.last_hidden_state,
                batch_dict["attention_mask"]
            )

            batch_embeddings = F.normalize(
                batch_embeddings,
                p=2,
                dim=1
            )

            embeddings.append(
                batch_embeddings
                .cpu()
                .numpy()
            )

    return np.vstack(embeddings)


# ============================================================
# 7. PREPARE LABELS
# ============================================================

le = LabelEncoder()

y = le.fit_transform(
    labels
)


# ============================================================
# 8. GENERATE ALL MESSAGE EMBEDDINGS
# ============================================================

X_embeddings = get_embeddings(
    messages,
    model,
    tokenizer,
    device
)


# ============================================================
# 9. CREATE METADATA
# ============================================================

metadata = [
    {
        "index": i,
        "message": message,
        "label": label,
        "label_encoded": y[i]
    }
    for i, (message, label)
    in enumerate(
        zip(messages, labels)
    )
]


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

TEST_SIZE = 0.1
SEED = 42

train_indices, test_indices = train_test_split(
    range(len(messages)),
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED
)


# ============================================================
# 11. SPLIT EMBEDDINGS + METADATA
# ============================================================

X_train_emb = X_embeddings[
    train_indices
]

X_test_emb = X_embeddings[
    test_indices
]

train_metadata = [
    metadata[i]
    for i in train_indices
]

test_metadata = [
    metadata[i]
    for i in test_indices
]


# ============================================================
# 12. CREATE FAISS INDEX
# ============================================================

embedding_dim = X_train_emb.shape[1]

index = faiss.IndexFlatIP(
    embedding_dim
)

index.add(
    X_train_emb.astype("float32")
)


# ============================================================
# 13. CLASSIFY WITH KNN
# ============================================================

def classify_with_knn(
    query_text,
    model,
    tokenizer,
    device,
    index,
    train_metadata,
    k=1
):
    """
    Classify text using k-nearest neighbors
    with embeddings.
    """

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_with_prefix = (
        f"query: {query_text}"
    )

    batch_dict = tokenizer(
        [query_with_prefix],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    batch_dict = {
        k: v.to(device)
        for k, v in batch_dict.items()
    }

    with torch.no_grad():

        outputs = model(
            **batch_dict
        )

        query_embedding = average_pool(
            outputs.last_hidden_state,
            batch_dict["attention_mask"]
        )

        query_embedding = F.normalize(
            query_embedding,
            p=2,
            dim=1
        )

        query_embedding = (
            query_embedding
            .cpu()
            .numpy()
            .astype("float32")
        )


    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        k
    )


    # --------------------------------------------------------
    # Get labels from top-k neighbors
    # --------------------------------------------------------

    predictions = []
    neighbor_info = []

    for i in range(k):

        neighbor_idx = (
            indices[0][i]
        )

        neighbor_score = (
            scores[0][i]
        )

        neighbor_label = (
            train_metadata[
                neighbor_idx
            ]["label"]
        )

        neighbor_message = (
            train_metadata[
                neighbor_idx
            ]["message"]
        )

        predictions.append(
            neighbor_label
        )

        neighbor_info.append(
            {
                "score": float(
                    neighbor_score
                ),

                "label": (
                    neighbor_label
                ),

                "message": (
                    neighbor_message[:100] + "..."
                    if len(neighbor_message) > 100
                    else neighbor_message
                )
            }
        )


    # --------------------------------------------------------
    # Majority vote
    # --------------------------------------------------------

    unique_labels, counts = np.unique(
        predictions,
        return_counts=True
    )

    final_prediction = (
        unique_labels[
            np.argmax(counts)
        ]
    )

    return (
        final_prediction,
        neighbor_info
    )


# ============================================================
# 14. EVALUATE KNN
# ============================================================

def evaluate_knn_accuracy(
    test_embeddings,
    test_labels,
    test_metadata,
    index,
    train_metadata,
    k_values=[1, 3, 5]
):
    """
    Evaluate accuracy for different k values
    using precomputed embeddings.
    """

    results = {}
    all_errors = {}

    for k in k_values:

        correct = 0
        total = len(
            test_embeddings
        )

        errors = []

        for i in tqdm(
            range(total),
            desc=f"Evaluating k={k}"
        ):

            query_embedding = (
                test_embeddings[
                    i:i + 1
                ]
                .astype("float32")
            )

            true_label = (
                test_metadata[i][
                    "label"
                ]
            )

            true_message = (
                test_metadata[i][
                    "message"
                ]
            )


            # -----------------------------------------------
            # Search FAISS
            # -----------------------------------------------

            scores, indices = index.search(
                query_embedding,
                k
            )


            # -----------------------------------------------
            # Get labels from neighbors
            # -----------------------------------------------

            predictions = []
            neighbor_details = []

            for j in range(k):

                neighbor_idx = (
                    indices[0][j]
                )

                neighbor_label = (
                    train_metadata[
                        neighbor_idx
                    ]["label"]
                )

                neighbor_message = (
                    train_metadata[
                        neighbor_idx
                    ]["message"]
                )

                neighbor_score = float(
                    scores[0][j]
                )

                predictions.append(
                    neighbor_label
                )

                neighbor_details.append(
                    {
                        "label":
                            neighbor_label,

                        "message":
                            neighbor_message,

                        "score":
                            neighbor_score
                    }
                )


            # -----------------------------------------------
            # Majority vote
            # -----------------------------------------------

            unique_labels, counts = np.unique(
                predictions,
                return_counts=True
            )

            predicted_label = (
                unique_labels[
                    np.argmax(counts)
                ]
            )

            if (
                predicted_label
                == true_label
            ):
                correct += 1

            else:

                error_info = {
                    "index":
                        i,

                    "original_index":
                        test_metadata[i][
                            "index"
                        ],

                    "message":
                        true_message,

                    "true_label":
                        true_label,

                    "predicted_label":
                        predicted_label,

                    "neighbors":
                        neighbor_details,

                    "label_distribution": {
                        label: int(count)

                        for label, count
                        in zip(
                            unique_labels,
                            counts
                        )
                    }
                }

                errors.append(
                    error_info
                )


        accuracy = (
            correct / total
        )

        error_count = (
            total - correct
        )

        results[k] = accuracy
        all_errors[k] = errors

        print(
            f"Accuracy with k={k}: "
            f"{accuracy:.4f}"
        )

        print(
            f"Number of errors with k={k}: "
            f"{error_count}/{total} "
            f"({(error_count / total) * 100:.2f}%)"
        )

    return (
        results,
        all_errors
    )


# ============================================================
# 15. EVALUATE TEST SET
# ============================================================

print(
    "Evaluating accuracy on test set..."
)

accuracy_results, error_results = (
    evaluate_knn_accuracy(
        X_test_emb,
        y_test,
        test_metadata,
        index,
        train_metadata,
        k_values=[1, 3, 5]
    )
)


# ============================================================
# 16. DISPLAY RESULTS
# ============================================================

print(
    "\n" + "=" * 50
)

print(
    "ACCURACY RESULTS"
)

print(
    "=" * 50
)

for k, accuracy in (
    accuracy_results.items()
):

    print(
        f"Top-{k} accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

print(
    "=" * 50
)


# ============================================================
# 17. SAVE ERROR ANALYSIS
# ============================================================

import json

from datetime import datetime


error_analysis = {

    "timestamp":
        datetime.now().isoformat(),

    "model":
        MODEL_NAME,

    "test_size":
        len(X_test_emb),

    "accuracy_results":
        accuracy_results,

    "errors_by_k":
        {}
}


for k, errors in (
    error_results.items()
):

    error_analysis[
        "errors_by_k"
    ][f"k_{k}"] = {

        "total_errors":
            len(errors),

        "error_rate":
            len(errors)
            / len(X_test_emb),

        "errors":
            errors
    }


output_file = (
    "error_analysis.json"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        error_analysis,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"\n***Error analysis saved to: "
    f"{output_file}***"
)

print()

print(
    "***Summary:"
)

for k, errors in (
    error_results.items()
):

    print(
        f"  k={k}: "
        f"{len(errors)} errors "
        f"out of "
        f"{len(X_test_emb)} samples"
    )


# ============================================================
# 18. COMPLETE CLASSIFICATION PIPELINE
# ============================================================

def spam_classifier_pipeline(
    user_input,
    k=3
):
    """
    Complete pipeline for spam classification.

    Args:
        user_input (str):
            Text to classify

        k (int):
            Number of nearest neighbors

    Returns:
        dict:
            Classification results
            with details
    """

    print()

    print(
        f'***Classifying: '
        f'"{user_input}"'
    )

    print()

    print(
        f"***Using top-{k} "
        f"nearest neighbors"
    )

    print()


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction, neighbors = (
        classify_with_knn(
            user_input,
            model,
            tokenizer,
            device,
            index,
            train_metadata,
            k=k
        )
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        f"***Prediction: "
        f"{prediction.upper()}"
    )

    print()

    print(
        "***Top neighbors:"
    )

    for i, neighbor in enumerate(
        neighbors,
        1
    ):

        print(
            f"{i}. Label: "
            f"{neighbor['label']} "
            f"| Score: "
            f"{neighbor['score']:.4f}"
        )

        print(
            f"   Message: "
            f"{neighbor['message']}"
        )

        print()


    # --------------------------------------------------------
    # Count labels
    # --------------------------------------------------------

    labels = [
        n["label"]
        for n in neighbors
    ]

    label_counts = {
        label:
            labels.count(label)

        for label
        in set(labels)
    }


    return {
        "prediction":
            prediction,

        "neighbors":
            neighbors,

        "label_distribution":
            label_counts
    }


# ============================================================
# 19. TEST PIPELINE
# ============================================================

test_examples = [

    "I am actually thinking a way "
    "of doing something useful",

    "FREE!! Click here to win "
    "$1000 NOW! Limited time offer!"
]


for i, example in enumerate(
    test_examples,
    1
):

    print(
        f'\n--- Example {i}: '
        f'"{example}" ---'
    )

    result = (
        spam_classifier_pipeline(
            example,
            k=3
        )
    )


# Interactive testing

print(
    "\n--- Interactive Testing ---"
)

user_text = (
    "Win a free iPhone! "
    "Click here now!"
)

k_value = 5

result = spam_classifier_pipeline(
    user_text,
    k=k_value
)