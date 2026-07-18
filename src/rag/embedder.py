import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

""" 
This file is the Brain Builder for your application. It contains the exact functions that turn the shredded text chunks we just talked about into mathematical coordinates, and it saves those coordinates onto your hard drive so the AI can search through them later.
"""


# 1. Grab the absolute path of your current working directory
# This ensures it always points to C:\Python_Study\BTP_1\faiss_index_db
DB_BASE_PATH = os.path.join(os.getcwd(), "faiss_index_db")

def get_db_path(session_id: str) -> str:
    """Helper to get the specific path for a session's DB."""
    return os.path.join(DB_BASE_PATH, session_id)

# ── Cached embedding model singleton ──
# HuggingFaceEmbeddings loads ~80MB of model weights from disk.
# Caching at module level avoids reloading on every vectorstore operation.
_embeddings = None

def get_embeddings():
    """Return the cached local embedding model (loaded once on first call)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def create_vectorstore(docs, session_id: str):
    """Build or extend the FAISS index for a session and save it to disk.

    If the session already has a vectorstore (e.g. a prior upload batch),
    the new chunks are merged into it so multi-document sessions accumulate
    instead of each upload call wiping out earlier documents.
    """
    print("Generating local embeddings and saving to disk...")
    embeddings = get_embeddings()
    existing = load_existing_vectorstore(session_id)
    if existing is not None:
        existing.add_documents(docs)
        vectorstore = existing
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)

    # Save the database locally using the absolute path
    db_path = get_db_path(session_id)
    os.makedirs(db_path, exist_ok=True)
    vectorstore.save_local(db_path)
    return vectorstore

def load_existing_vectorstore(session_id: str):
    """Check if a database already exists on startup and load it."""
    db_path = get_db_path(session_id)
    # Check if the index file actually exists inside our folder
    index_file = os.path.join(db_path, "index.faiss")
    
    if os.path.exists(db_path) and os.path.exists(index_file):
        embeddings = get_embeddings()
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return None