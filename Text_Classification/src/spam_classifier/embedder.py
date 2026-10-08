import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from spam_classifier.config import MODEL_NAME, BATCH_SIZE


class E5Embedder:

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.device = self._get_device()

        print(f"Loading embedding model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Device: {self.device}")

    @staticmethod
    def _get_device():
        """Select the best available device."""

        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    @staticmethod
    def average_pool(last_hidden_states, attention_mask):
        """Convert token embeddings into one sentence embedding."""

        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(),
            0.0
        )

        return (
            last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        )

    def embed_batch(self, texts, prefix="passage"):
        """Generate embeddings for one batch of texts."""

        prefixed_texts = [
            f"{prefix}: {text}"
            for text in texts
        ]

        batch_dict = self.tokenizer(
            prefixed_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        batch_dict = {
            key: value.to(self.device)
            for key, value in batch_dict.items()
        }

        with torch.no_grad():
            outputs = self.model(**batch_dict)

            embeddings = self.average_pool(
                outputs.last_hidden_state,
                batch_dict["attention_mask"]
            )

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

        return embeddings.cpu().numpy().astype("float32")

    def get_embeddings(self, texts, batch_size=BATCH_SIZE, prefix="passage"):
        """Generate embeddings for multiple texts."""

        embeddings = []

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Generating embeddings"
        ):
            batch_texts = texts[i:i + batch_size]

            batch_embeddings = self.embed_batch(
                batch_texts,
                prefix=prefix
            )

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)