import os
from typing import List

from sentence_transformers import SentenceTransformer

CACHE_DIR = ".hf_cache"


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 48):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HOME", CACHE_DIR)
        os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
        self.model = SentenceTransformer(model_name, cache_folder=CACHE_DIR)
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed list of texts in batches.
        Returns list of vectors aligned with input order.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
