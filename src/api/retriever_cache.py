"""Process-local LRU cache of per-session hybrid retrievers.

Building a retriever means loading a FAISS index from disk and fitting a TF-IDF
model over its docstore, which is far too slow to redo on every chat turn.
Entries are evicted least-recently-used first because each one pins a FAISS
index in RAM (~10-50MB).
"""
from collections import OrderedDict
from typing import Optional

from langchain_community.retrievers import TFIDFRetriever

from src.rag.embedder import load_existing_vectorstore
from src.rag.retriever import HybridRetriever

_cache: OrderedDict = OrderedDict()
_MAX_ENTRIES = 32
_TOP_N = 4


def _build(vectorstore) -> HybridRetriever:
    """Fit a sparse TF-IDF retriever over the vectorstore's docs and pair the two."""
    docs = list(vectorstore.docstore._dict.values())
    tfidf_retriever = TFIDFRetriever.from_documents(docs)
    return HybridRetriever(vectorstore=vectorstore, tfidf_retriever=tfidf_retriever, top_n=_TOP_N)


def get(session_id: str) -> Optional[HybridRetriever]:
    """Return a cached retriever, loading from disk if needed. None if no vectorstore exists."""
    if session_id in _cache:
        _cache.move_to_end(session_id)  # Mark as recently used
        return _cache[session_id]

    db = load_existing_vectorstore(session_id)
    if not db:
        return None

    retriever = _build(db)
    _cache[session_id] = retriever
    if len(_cache) > _MAX_ENTRIES:
        _cache.popitem(last=False)  # Evict least-recently-used
    return retriever


def refresh(session_id: str, vectorstore) -> None:
    """Rebuild and cache the retriever after a session's vectorstore changes."""
    _cache[session_id] = _build(vectorstore)


def invalidate(session_id: str) -> None:
    """Drop a session's cached retriever, if present."""
    _cache.pop(session_id, None)
