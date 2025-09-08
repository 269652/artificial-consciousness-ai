"""Lightweight QA Layer

Goals:
- Small local instruct model (default: microsoft/Phi-3-mini-4k-instruct if available)
- Embedding-based retrieval (sentence-transformers/all-MiniLM-L6-v2 fallback to hash)
- Simple RAG prompt assembly

Usage:
    qa = QALayer()
    qa.add_documents([{"id":"doc1","text":"The sky is blue."}])
    answer = qa.answer("What color is the sky?")

Environment overrides:
    QA_GEN_MODEL   -> HF model id for generation
    QA_EMB_MODEL   -> HF model id for embeddings
    QA_MAX_DOCS    -> limit documents kept (FIFO)
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Sequence
import os
import math
import logging
import threading
import heapq
import random
import re

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    _HF_AVAIL = True
except Exception:
    _HF_AVAIL = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAIL = True
except Exception:
    _ST_AVAIL = False

_DEF_GEN_MODEL_CANDIDATES = [
    os.environ.get("QA_GEN_MODEL"),
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]
_DEF_EMB_MODEL_CANDIDATES = [
    os.environ.get("QA_EMB_MODEL"),
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
]

def _hash_vec(text: str, dim: int = 384) -> List[float]:
    random.seed(hash(text) & 0xffffffff)
    return [random.random() for _ in range(dim)]

class _EmbeddingIndex:
    def __init__(self, model_id: Optional[str] = None):
        self._model = None
        self.dim = 0
        if _ST_AVAIL:
            chosen = None
            for cand in _DEF_EMB_MODEL_CANDIDATES:
                if cand:
                    try:
                        self._model = SentenceTransformer(cand)
                        self.dim = self._model.get_sentence_embedding_dimension()
                        chosen = cand
                        logger.info("Embedding model loaded: %s", cand)
                        break
                    except Exception as e:  # pragma: no cover
                        logger.warning("Embedding model load failed %s: %s", cand, e)
            if chosen is None:
                logger.warning("Falling back to hash embeddings")
        else:
            logger.warning("sentence-transformers unavailable; using hash embeddings")
        if self.dim == 0:
            self.dim = 384
        self._lock = threading.Lock()
        self.docs: List[Dict[str, Any]] = []  # {id,text,vec}
        self.max_docs = int(os.environ.get("QA_MAX_DOCS", "10000"))

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        if self._model is None:
            return [_hash_vec(t, self.dim) for t in texts]
        try:
            embs = self._model.encode(list(texts))  # type: ignore
            return [list(e) for e in embs]
        except Exception as e:
            logger.warning("Embedding encode failed (%s); falling back to hash", e)
            return [_hash_vec(t, self.dim) for t in texts]

    def add(self, docs: List[Dict[str, str]]):
        with self._lock:
            vecs = self._embed([d["text"] for d in docs])
            for d, v in zip(docs, vecs):
                self.docs.append({"id": d["id"], "text": d["text"], "vec": v})
            if len(self.docs) > self.max_docs:
                excess = len(self.docs) - self.max_docs
                if excess > 0:
                    self.docs = self.docs[excess:]

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.docs:
            return []
        qv = self._embed([text])[0]
        # Cosine similarity
        def cos(a, b):
            dot = sum(x*y for x,y in zip(a,b))
            na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
            if na == 0 or nb == 0: return 0.0
            return dot/(na*nb)
        heap = []
        for d in self.docs:
            s = cos(qv, d["vec"])
            if len(heap) < k:
                heapq.heappush(heap, (s, d))
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (s, d))
        return [d for _, d in sorted(heap, key=lambda x: x[0], reverse=True)]

class QALayer:
    def __init__(self):
        self.index = _EmbeddingIndex()
        self.gen_pipe = None
        if _HF_AVAIL:
            for cand in _DEF_GEN_MODEL_CANDIDATES:
                if not cand: continue
                try:
                    self.gen_pipe = pipeline("text-generation", model=cand, trust_remote_code=True, device_map="auto")
                    logger.info("Loaded generation model: %s", cand)
                    break
                except Exception as e:  # pragma: no cover
                    logger.warning("Gen model load failed %s: %s", cand, e)
        if self.gen_pipe is None:
            logger.warning("No generation model loaded; answers will be heuristic")

    def add_documents(self, docs: List[Dict[str, str]]):
        # docs: [{id,text}]
        self.index.add(docs)

    def _clean(self, q: str) -> str:
        q = q.strip()
        q = re.sub(r"\s+", " ", q)
        return q.rstrip("?!. ")

    def _assemble_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        ctx_block = "\n".join(f"[Doc {i+1}] {c['text']}" for i, c in enumerate(contexts))
        return (
            "You are a concise factual QA assistant. Answer using ONLY the provided context. "
            "If the answer is absent, say 'Insufficient information.'\n" +
            "\nContext:\n" + ctx_block + "\n\nQuestion: " + question + "\nAnswer:" )

    def answer(self, question: str, top_k: int = 5, max_new_tokens: int = 128) -> Dict[str, Any]:
        q_clean = self._clean(question)
        contexts = self.index.query(q_clean, k=top_k)
        prompt = self._assemble_prompt(q_clean, contexts)
        if self.gen_pipe:
            try:
                out = self.gen_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.7)
                if out and isinstance(out, list):
                    text = out[0]["generated_text"]
                    # Extract answer after last 'Answer:'
                    ans = text.split("Answer:")[-1].strip()
                else:
                    ans = "(no output)"
            except Exception as e:
                logger.warning("Generation error: %s", e)
                ans = "(generation failed)"
        else:
            # Heuristic answer: pick sentence with most overlapping words
            if contexts:
                q_words = set(w.lower() for w in re.findall(r"\w+", q_clean))
                best = max(contexts, key=lambda c: len(q_words & set(re.findall(r"\w+", c['text'].lower()))))
                ans = best["text"]
            else:
                ans = "Insufficient information."
        return {"question": question, "answer": ans, "contexts": contexts, "prompt": prompt}

if __name__ == "__main__":
    print("HEy")
    logging.basicConfig(level=logging.INFO)
    qa = QALayer()
    # qa.add_documents([
    #     {"id": "d1", "text": "Paris is the capital of France."},
    #     {"id": "d2", "text": "Berlin is the capital of Germany."},
    #     {"id": "d3", "text": "The Eiffel Tower is located in Paris."},
    # ])
    print(qa.answer("Where is the Eiffel Tower?"))
