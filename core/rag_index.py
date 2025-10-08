# core/rag_index.py
from __future__ import annotations
from typing import List, Tuple
import os, re, json
from pathlib import Path

# --- CPU-friendly embeddings ---
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "data/rag_index"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def _clean_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', s or '').strip()
    return s

def chunk_text(text: str, max_tokens: int = 800, overlap: int = 120) -> List[str]:
    # crude tokenization by words; keeps it simple & fast
    words = _clean_text(text).split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        end = min(i + max_tokens, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words): break
        i = max(0, end - overlap)
    return chunks

class LocalFaissIndex:
    """
    Minimal FAISS index wrapper storing:
      - faiss index (L2)
      - np.float32 embeddings
      - list[str] texts
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts: List[str] = []
        self.dim = self.model.get_sentence_embedding_dimension()

    def add_texts(self, texts: List[str]):
        texts = [t for t in (texts or []) if _clean_text(t)]
        if not texts: return
        embs = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embs = np.asarray(embs, dtype="float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)  # cosine via dot on normalized vecs
        self.index.add(embs)
        self.texts.extend(texts)

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if not query or self.index is None or len(self.texts) == 0:
            return []
        q = self.model.encode([query], show_progress_bar=False, normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q, k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            out.append((self.texts[idx], float(score)))
        return out

    # ---------- persistence ----------
    def save(self, path: Path = INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)
        # save FAISS
        faiss.write_index(self.index, str(path / "faiss.index"))
        # save texts
        (path / "texts.json").write_text(json.dumps(self.texts, ensure_ascii=False))
        # save meta
        (path / "meta.json").write_text(json.dumps({"model": MODEL_NAME, "dim": self.dim}))

    @classmethod
    def load(cls, path: Path = INDEX_DIR) -> "LocalFaissIndex|None":
        try:
            meta = json.loads((path / "meta.json").read_text())
            inst = cls(model_name=meta.get("model", MODEL_NAME))
            inst.index = faiss.read_index(str(path / "faiss.index"))
            inst.texts = json.loads((path / "texts.json").read_text())
            return inst
        except Exception:
            return None

def build_or_refresh_index(resume_text: str, extra_docs: List[str] | None = None) -> LocalFaissIndex:
    idx = LocalFaissIndex()
    chunks = chunk_text(resume_text or "")
    if extra_docs:
        for doc in extra_docs:
            chunks.extend(chunk_text(doc or ""))
    idx.add_texts(chunks)
    idx.save(INDEX_DIR)
    return idx
