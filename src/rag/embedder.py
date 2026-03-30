import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Grab the absolute path of your current working directory
# This ensures it always points to C:\Python_Study\BTP_1\faiss_index_db
DB_BASE_PATH = os.path.join(os.getcwd(), "faiss_index_db")

def get_db_path(session_id: str) -> str:
    """Helper to get the specific path for a session's DB."""
    return os.path.join(DB_BASE_PATH, session_id)

def get_embeddings():
    """Load the fast local embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vectorstore(docs, session_id: str):
    """Build FAISS index and save it permanently to the hard drive for the session."""
    print("Generating local embeddings and saving to disk...")
    embeddings = get_embeddings()
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