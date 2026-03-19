from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.core.config import GOOGLE_API_KEY

def create_vectorstore(docs):
    """Build FAISS index with Gemini embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
    return FAISS.from_documents(docs, embeddings)