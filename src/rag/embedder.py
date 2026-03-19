# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from src.core.config import GOOGLE_API_KEY

# def create_vectorstore(docs):
#     """Build FAISS index with Gemini embeddings."""
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
#     return FAISS.from_documents(docs, embeddings)

# import time
# import streamlit as st
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from src.core.config import GOOGLE_API_KEY

# def create_vectorstore(docs):
#     """Build FAISS index with Gemini embeddings, avoiding 429 Rate Limits."""
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001", 
#         google_api_key=GOOGLE_API_KEY
#     )
    
#     # If the file is small enough, just process it normally
#     batch_size = 50 
#     if len(docs) <= batch_size:
#         return FAISS.from_documents(docs, embeddings)
    
#     # Otherwise, we batch the processing to respect the 100 RPM limit
#     st.info(f"Large document detected ({len(docs)} chunks). Processing in batches to avoid API limits...")
    
#     # 1. Create the initial vectorstore with the first batch
#     vectorstore = FAISS.from_documents(docs[:batch_size], embeddings)
    
#     # 2. Add the remaining documents in batches with a delay
#     progress_bar = st.progress(batch_size / len(docs))
    
#     for i in range(batch_size, len(docs), batch_size):
#         # Sleep for 30 seconds to let the rate limit reset
#         time.sleep(30) 
        
#         # Process the next batch
#         batch = docs[i : i + batch_size]
#         vectorstore.add_documents(batch)
        
#         # Update the progress bar in the UI
#         current_progress = min((i + batch_size) / len(docs), 1.0)
#         progress_bar.progress(current_progress)
        
#     return vectorstore


# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# def create_vectorstore(docs):
#     """Build FAISS index with local HuggingFace embeddings (NO RATE LIMITS!)."""
    
#     st.info("Generating local embeddings (this is fast and has no rate limits)...")
    
#     # Load the local open-source embedding model
#     # The model will download once on the first run, then load instantly from cache
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
#     # Build and return the vectorstore instantly without batching
#     return FAISS.from_documents(docs, embeddings)


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