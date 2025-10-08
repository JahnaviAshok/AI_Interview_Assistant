# core/rag_retriever.py
from __future__ import annotations
from typing import List
from pathlib import Path
from .rag_index import LocalFaissIndex, INDEX_DIR

def get_index() -> LocalFaissIndex | None:
    return LocalFaissIndex.load(INDEX_DIR)

def retrieve_context(query: str, k: int = 3) -> List[str]:
    idx = get_index()
    if not idx:
        return []
    results = idx.search(query, k=k)
    return [t for t, _ in results]
