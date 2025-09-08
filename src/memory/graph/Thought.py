"""Thought representation with optional T5-based embedding.
Falls back to a simple hash vector if sentence-transformers / model unavailable.
"""
from __future__ import annotations
from typing import List, Dict, Optional
import logging
import math
import random

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_NAME_CANDIDATES = [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2", 
    "all-mpnet-base-v2",
    "paraphrase-albert-small-v2",
]


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.warning("sentence-transformers not available (%s); using hash embeddings", e)
        _MODEL = None
        return _MODEL
    for name in _MODEL_NAME_CANDIDATES:
        try:
            _MODEL = SentenceTransformer(name)
            logger.info("Loaded embedding model: %s", name)
            return _MODEL
        except Exception as e:  # pragma: no cover
            logger.warning("Failed loading model %s: %s", name, e)
    logger.warning("All model load attempts failed; using hash embeddings")
    return _MODEL


def _hash_embed(text: str, dim: int = 64) -> List[float]:
    random.seed(hash(text) & 0xffffffff)
    return [random.random() for _ in range(dim)]


class Thought:
    def __init__(self, text: str, embedding: List[float], scores: Optional[Dict[str, float]] = None):
        self.text = text
        self.embedding = embedding
        self.scores: Dict[str, float] = scores if scores is not None else {}

    @classmethod
    def from_text(cls, text: str) -> "Thought":
        model = _load_model()
        if model is None:
            emb = _hash_embed(text)
        else:
            try:
                emb = list(model.encode([text])[0])  # type: ignore
            except Exception as e:  # pragma: no cover
                logger.warning("Model encode failed (%s); falling back to hash", e)
                emb = _hash_embed(text)
        return cls(text=text, embedding=emb, scores={})

    @classmethod
    def from_texts(cls, texts: List[str], dim: int = 64) -> 'Thought':
        """Create a Thought from multiple texts by averaging their embeddings"""
        if not texts:
            return cls("", _hash_embed("", dim))
        
        model = _load_model()
        if model is not None:
            try:
                embeddings = model.encode(texts)
                # Average the embeddings
                avg_embedding = embeddings.mean(axis=0).tolist()
                combined_text = " | ".join(texts)
                return cls(combined_text, avg_embedding)
            except Exception as e:
                logger.warning("Model encoding failed (%s); using hash embeddings", e)
        
        # Fallback: hash each text and average
        hash_embeddings = [_hash_embed(text, dim) for text in texts]
        avg_embedding = [sum(vals) / len(vals) for vals in zip(*hash_embeddings)]
        combined_text = " | ".join(texts)
        return cls(combined_text, avg_embedding)

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.embedding))

    def __repr__(self):  # pragma: no cover
        return f"Thought(len_text={len(self.text)}, dim={len(self.embedding)})"
