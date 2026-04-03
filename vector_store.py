"""
Vector Store Module
Embeds text chunks using TF-IDF (offline, no model download) with
cosine similarity retrieval. Optionally upgrades to sentence-transformers
when HuggingFace is reachable for higher semantic quality.

Supports save/load so the index persists between runs.
"""

import os
import pickle
from typing import List, Dict

import numpy as np


# ------------------------------------------------------------------ #
#  Embedding backends                                                  #
# ------------------------------------------------------------------ #

class _TFIDFBackend:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )

    def fit(self, texts: List[str]):
        self._vectorizer.fit(texts)

    def embed(self, texts: List[str]) -> np.ndarray:
        mat = self._vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms


def _try_sentence_transformers(model_name: str):
    """Return (backend_instance, True) or (None, False)."""
    try:
        from sentence_transformers import SentenceTransformer

        class _ST:
            name = "sentence-transformers"
            def __init__(self, enc): self._enc = enc
            def embed(self, texts):
                v = self._enc.encode(texts, convert_to_numpy=True,
                                     normalize_embeddings=True)
                return v.astype("float32")

        enc = SentenceTransformer(model_name)
        return _ST(enc), True
    except Exception:
        return None, False


# ------------------------------------------------------------------ #
#  VectorStore                                                         #
# ------------------------------------------------------------------ #

class VectorStore:
    """
    Lightweight cosine-similarity vector store.

    Embedding priority:
      1. sentence-transformers (best quality, needs HuggingFace)
      2. TF-IDF bi-gram (offline, always available)
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DATA_FILE = "vector_store.pkl"

    def __init__(self, index_path: str = "./vector_index"):
        self.index_path = index_path
        self._embeddings: np.ndarray | None = None
        self._metadata: List[Dict] = []
        self._backend = None
        self._backend_name: str = ""

    def build(self, chunks: List[Dict]) -> None:
        texts = [c["text"] for c in chunks]

        st, ok = _try_sentence_transformers(self.EMBEDDING_MODEL)
        if ok:
            self._backend = st
            print("[VectorStore] Using sentence-transformers backend.")
        else:
            print("[VectorStore] Using TF-IDF backend (offline mode).")
            tfidf = _TFIDFBackend()
            tfidf.fit(texts)
            self._backend = tfidf

        self._backend_name = self._backend.name
        self._embeddings = self._backend.embed(texts)
        self._metadata = list(chunks)

    def save(self) -> None:
        os.makedirs(self.index_path, exist_ok=True)
        with open(os.path.join(self.index_path, self.DATA_FILE), "wb") as f:
            pickle.dump({
                "embeddings":   self._embeddings,
                "metadata":     self._metadata,
                "backend":      self._backend,
                "backend_name": self._backend_name,
            }, f)
        print(f"[VectorStore] Saved {len(self._metadata)} chunks → '{self.index_path}'.")

    def load(self) -> bool:
        path = os.path.join(self.index_path, self.DATA_FILE)
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            p = pickle.load(f)
        self._embeddings   = p["embeddings"]
        self._metadata     = p["metadata"]
        self._backend      = p["backend"]
        self._backend_name = p.get("backend_name", "unknown")
        return True

    def size(self) -> int:
        return len(self._metadata)

    def search(self, query: str, k: int = 6) -> List[Dict]:
        if self._embeddings is None or not self._metadata:
            return []
        q_vec = self._backend.embed([query])
        scores = (self._embeddings @ q_vec.T).flatten()
        k = min(k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        results = []
        for idx in top_idx:
            chunk = dict(self._metadata[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results
