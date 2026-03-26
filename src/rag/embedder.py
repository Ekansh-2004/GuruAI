
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Grab the absolute path of your current working directory
# This ensures it always points to C:\Python_Study\BTP_1\vector_db
DB_PATH = os.path.join(os.getcwd(), "faiss_index_db")

# 2. Force create the directory if it doesn't exist yet
os.makedirs(DB_PATH, exist_ok=True)

def get_embeddings():
    """Load the fast local embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vectorstore(docs):
    """Build FAISS index and save it permanently to the hard drive."""
    st.info("Generating local embeddings and saving to disk...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save the database locally using the absolute path
    vectorstore.save_local(DB_PATH)
    return vectorstore

def load_existing_vectorstore():
    """Check if a database already exists on startup and load it."""
    # Check if the index file actually exists inside our folder
    index_file = os.path.join(DB_PATH, "index.faiss")
    
    if os.path.exists(DB_PATH) and os.path.exists(index_file):
        embeddings = get_embeddings()
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None