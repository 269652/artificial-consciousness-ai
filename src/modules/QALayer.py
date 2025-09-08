"""Lightweight QA Layer

Goals:
- Small local instruct model (default: microsoft/Phi-3-mini-4k-instruct if available)
- Embedding-based retrieval (sentence-transformers/all-MiniLM-L6-v2 fallback to hash)
- Simple RAG prompt assembly
- Optional external LLM (OpenAI / Anthropic) fallback for broad pretrained knowledge

Env Vars:
    QA_GEN_MODEL       local HF model id override
    QA_EMB_MODEL       embedding model override
    QA_MAX_DOCS        int limit
    QA_PROVIDER        one of: local, openai, anthropic (auto if unset)
    OPENAI_API_KEY     key for OpenAI
    OPENAI_MODEL       e.g. gpt-4o-mini / gpt-4-turbo / gpt-3.5-turbo
    ANTHROPIC_API_KEY  key for Anthropic
    ANTHROPIC_MODEL    e.g. claude-3-5-sonnet-latest
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
import json
import time
import urllib.request
import urllib.error

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

# External provider availability (lazy HTTP usage; avoid mandatory SDKs)
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")

_DEF_GEN_MODEL_CANDIDATES = [

    "google/flan-t5-large",
]

# Approx parameter counts (billions)
_MODEL_PARAM_B = {
    "google/flan-t5-large": 3,
    "microsoft/phi-3-mini-4k-instruct": 3.8,
    "microsoft/phi-3.5-mini-instruct": 3.8,
    "qwen/qwen2-3b-instruct": 3.0,
    "qwen/qwen2.5-3b-instruct": 3.0,
    "google/gemma-2-2b-it": 2.0,
    "meta-llama/meta-llama-3-8b-instruct": 8.0,
    "mistralai/mistral-7b-instruct-v0.3": 7.0,
    "qwen/qwen2-7b-instruct": 7.0,
    "tinyllama/tinyllama-1.1b-chat-v1.0": 1.1,
    "t5-small": 0.06,
}

# Embedding model candidates (restored after refactor)
_DEF_EMB_MODEL_CANDIDATES = [
    "google/flan-t5-large",
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

# ========== External LLM helper functions ==========

def _openai_chat(prompt: str, max_new_tokens: int = 256) -> Optional[str]:
    if not _OPENAI_KEY:
        return None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {_OPENAI_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise factual QA assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.2
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        choices = data.get("choices") or []
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
    except urllib.error.HTTPError as e:
        logger.warning("OpenAI HTTP error %s", e.read())
    except Exception as e:  # pragma: no cover
        logger.warning("OpenAI request failed: %s", e)
    return None

def _anthropic_chat(prompt: str, max_new_tokens: int = 256) -> Optional[str]:
    if not _ANTHROPIC_KEY:
        return None
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": _ANTHROPIC_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": model,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "system": "You are a concise factual QA assistant.",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        content = data.get("content") or []
        if content and isinstance(content, list):
            # Each item may be {type: "text", text: "..."}
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    return c.get("text", "").strip()
    except urllib.error.HTTPError as e:
        logger.warning("Anthropic HTTP error %s", e.read())
    except Exception as e:  # pragma: no cover
        logger.warning("Anthropic request failed: %s", e)
    return None

# ========== Embedding index (unchanged logic) ==========
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

# ========== QALayer ==========
class QALayer:
    def __init__(self):
        self.index = _EmbeddingIndex()
        self.gen_pipe = None
        self.provider = (os.environ.get("QA_PROVIDER") or "auto").lower()
        self._local_enabled = False
        self._is_text2text = False
        load_4bit = os.environ.get("QA_LOAD_4BIT", "1") in ("1", "true", "True")
        min_params = float(os.environ.get("QA_MIN_PARAMS", "3"))  # billions
        # Local model selection
        if self.provider in ("auto", "local") and _HF_AVAIL:
            for cand in _DEF_GEN_MODEL_CANDIDATES:
                if not cand:
                    continue
                key = cand.lower()
                pcount = _MODEL_PARAM_B.get(key, 0.0)
                if pcount < min_params:
                    continue  # enforce minimum size
                try:
                    if load_4bit:
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        tok = AutoTokenizer.from_pretrained(cand, trust_remote_code=True)
                        # Decide text2text vs causal: T5 family uses text2text pipeline better
                        is_t2t = 't5' in cand.lower()
                        if is_t2t:
                            from transformers import AutoModelForSeq2SeqLM
                            model = AutoModelForSeq2SeqLM.from_pretrained(
                                cand,
                                trust_remote_code=True,
                                device_map="auto",
                                load_in_8bit=False,
                            )
                            from transformers import pipeline as _pl
                            self.gen_pipe = _pl("text2text-generation", model=model, tokenizer=tok, device_map="auto")
                            self._is_text2text = True
                        else:
                            model = AutoModelForCausalLM.from_pretrained(
                                cand,
                                trust_remote_code=True,
                                device_map="auto",
                                load_in_4bit=True,
                            )
                            from transformers import TextGenerationPipeline
                            self.gen_pipe = TextGenerationPipeline(model=model, tokenizer=tok)
                    else:
                        if 't5' in cand.lower():
                            self.gen_pipe = pipeline("text2text-generation", model=cand, trust_remote_code=True, device_map="auto")
                            self._is_text2text = True
                        else:
                            self.gen_pipe = pipeline("text-generation", model=cand, trust_remote_code=True, device_map="auto")
                    self._local_enabled = True
                    logger.info("Loaded local generation model: %s (params>=%.1fB, 4bit=%s)", cand, min_params, load_4bit)
                    break
                except Exception as e:  # pragma: no cover
                    logger.warning("Local gen model load failed %s: %s", cand, e)
        if not self.gen_pipe and self.provider == "local":
            logger.warning("Provider forced to local but no model meeting QA_MIN_PARAMS=%s was loaded.", min_params)
        if self.provider == "openai" and not _OPENAI_KEY:
            logger.warning("QA_PROVIDER=openai but OPENAI_API_KEY missing.")
        if self.provider == "anthropic" and not _ANTHROPIC_KEY:
            logger.warning("QA_PROVIDER=anthropic but ANTHROPIC_API_KEY missing.")
        if self.provider == "auto" and not self.gen_pipe:
            if _OPENAI_KEY:
                self.provider = "openai"
            elif _ANTHROPIC_KEY:
                self.provider = "anthropic"
            else:
                self.provider = "heuristic"
        if not self.gen_pipe and self.provider not in ("openai", "anthropic", "heuristic"):
            logger.info("Falling back to heuristic mode")
            self.provider = "heuristic"

    def add_documents(self, docs: List[Dict[str, str]]):
        # docs: [{id,text}]
        self.index.add(docs)

    def _clean(self, q: str) -> str:
        q = q.strip()
        q = re.sub(r"\s+", " ", q)
        return q.rstrip("?!. ")

    def _assemble_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        if contexts:
            ctx_block = "\n".join(f"[Doc {i+1}] {c['text']}" for i, c in enumerate(contexts))
            return (
                "You are a concise factual QA assistant. Answer using ONLY the provided context. "
                "If the answer is absent, say 'Insufficient information.'\n" +
                "\nContext:\n" + ctx_block + "\n\nQuestion: " + question + "\nAnswer:")
        else:
            # Open-domain prompt when no retrieved docs
            return (
                "You are a knowledgeable, concise assistant. Provide an accurate factual answer. "
                "If the question is ambiguous, state the ambiguity.\nQuestion: " + question + "\nAnswer:")

    def answer(self, question: str, top_k: int = 5, max_new_tokens: int = 128) -> Dict[str, Any]:
        q_clean = self._clean(question)
        contexts = self.index.query(q_clean, k=top_k)
        prompt = self._assemble_prompt(q_clean, contexts)
        ans: str
        if self.provider == "openai":
            ans = _openai_chat(prompt, max_new_tokens) or "(no response)"
        elif self.provider == "anthropic":
            ans = _anthropic_chat(prompt, max_new_tokens) or "(no response)"
        elif self.gen_pipe:
            try:
                if self._is_text2text:
                    out = self.gen_pipe(prompt, max_new_tokens=max_new_tokens)
                    if out and isinstance(out, list):
                        ans = out[0].get("generated_text", "").strip()
                    else:
                        ans = "(no output)"
                else:
                    out = self.gen_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.2)
                    if out and isinstance(out, list):
                        text = out[0].get("generated_text", "")
                        ans = text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()
                    else:
                        ans = "(no output)"
            except Exception as e:
                logger.warning("Local generation error: %s", e)
                ans = "(generation failed)"
        else:
            if contexts:
                q_words = set(w.lower() for w in re.findall(r"\w+", q_clean))
                best = max(contexts, key=lambda c: len(q_words & set(re.findall(r"\w+", c['text'].lower()))))
                ans = best["text"]
            else:
                ans = "Insufficient information."
        return {"question": question, "answer": ans, "provider": self.provider, "contexts": contexts, "prompt": prompt}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    qa = QALayer()
    # qa.add_documents([
    #     {"id": "d1", "text": "Paris is the capital of France."},
    #     {"id": "d2", "text": "Berlin is the capital of Germany."},
    #     {"id": "d3", "text": "The Eiffel Tower is located in Paris."},
    # ])
    print(qa.answer("Where is the Eiffel Tower?"))
