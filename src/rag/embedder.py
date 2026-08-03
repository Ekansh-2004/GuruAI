import json

from langchain_huggingface import HuggingFaceEmbeddings
from pgvector import Vector

from src.core.database import get_db

"""
This file is the Brain Builder for your application. It embeds each uploaded
chunk and stores it in the `document_chunks` table (Postgres + pgvector),
keyed by session_id, so retrieval can query it directly instead of loading a
local FAISS index.
"""

_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Cached embedding model singleton ──
# HuggingFaceEmbeddings loads ~80MB of model weights from disk.
# Caching at module level avoids reloading on every embed call.
_embeddings = None

def get_embeddings():
    """Return the cached local embedding model (loaded once on first call)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=_MODEL_NAME)
    return _embeddings

def get_db_path(session_id: str) -> str:
    """Legacy label, kept only for the `documents.storage_path` metadata column.
    Chunks now live in the document_chunks table, not on local disk."""
    return f"postgres:document_chunks:{session_id}"

def vectorstore_exists(session_id: str) -> bool:
    """True if a session has any embedded chunks stored."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM document_chunks WHERE session_id = %s LIMIT 1", (session_id,))
        return cur.fetchone() is not None

def delete_vectorstore(session_id: str) -> None:
    """Remove a session's embedded chunks. No-op if there were none."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM document_chunks WHERE session_id = %s", (session_id,))
        conn.commit()

def create_vectorstore(docs, session_id: str):
    """Embed each chunk and insert it into document_chunks for this session.

    Multi-document sessions accumulate naturally: this always inserts new rows
    rather than replacing existing ones, so a second upload batch adds to the
    first instead of wiping it out.
    """
    print("Generating local embeddings and saving to Postgres...")
    model = get_embeddings()
    texts = [d.page_content for d in docs]
    vectors = model.embed_documents(texts)

    with get_db() as conn:
        cur = conn.cursor()
        for doc, vector in zip(docs, vectors):
            cur.execute(
                """
                INSERT INTO document_chunks (session_id, document_id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    doc.metadata.get("document_id"),
                    doc.page_content,
                    json.dumps(doc.metadata),
                    Vector(vector),
                ),
            )
        conn.commit()
    return session_id

def load_existing_vectorstore(session_id: str):
    """Kept for interface compatibility with older callers; returns the
    session_id if it has embedded chunks, else None. New code should query
    document_chunks directly (see src/rag/retriever.py, src/api/retriever_cache.py,
    src/rag/quiz.py) rather than treating this as a loadable FAISS object."""
    return session_id if vectorstore_exists(session_id) else None
