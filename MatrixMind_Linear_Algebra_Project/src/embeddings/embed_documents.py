"""Small, deterministic local text embeddings for MatrixMind.

Uses feature hashing (unigrams, bigrams and character n-grams), so indexing and
retrieval work offline without downloading a transformer model. The vectors
are intentionally lightweight; Groq is optional for answer generation.
"""
from __future__ import annotations

import hashlib
import re
import numpy as np

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+(?:\.\d+)?")


class EmbeddingEngine:
    def __init__(self, model_name: str | None = None, dimensions: int = 384):
        self.model_name = model_name or "local-feature-hashing-v1"
        self.dimensions = dimensions

    def _embed_one(self, text: str) -> np.ndarray:
        text = (text or "").casefold()
        tokens = TOKEN_RE.findall(text)
        features: list[tuple[str, float]] = []
        features.extend((f"w:{token}", 1.0) for token in tokens)
        features.extend((f"b:{a}_{b}", 0.72) for a, b in zip(tokens, tokens[1:]))
        compact = re.sub(r"\s+", " ", text)
        for size in (3, 4):
            features.extend((f"c{size}:{compact[i:i+size]}", 0.16) for i in range(max(0, len(compact)-size+1)))
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed an empty list of texts")
        return np.vstack([self._embed_one(text) for text in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_one(query)

    def get_embedding_dim(self) -> int:
        return self.dimensions
