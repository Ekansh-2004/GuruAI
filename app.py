from src.rag.loader import load_documents
from src.rag.embedder import create_vectorstore
from src.rag.chain import build_rag_chain
from src.personalization.tracker import (
    load_history, add_message, clear_history, get_recent_weak_topics
)
import streamlit as st
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads .env from project root
except ImportError:
    st.error("pip install python-dotenv")
    st.stop()

# Set API key from .env ONLY (no st.secrets)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Add GOOGLE_API_KEY=your_key to .env file in project root!")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ... rest of your imports/app
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.stApp { background: #0d0d0d; color: #e8e8e8; }
.hero { padding: 2.5rem 0 1.5rem 0; text-align: center; }
.hero h1 {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.8rem;
    color: #f5f5f5; letter-spacing: -1px; margin-bottom: 0.3rem;
}
.hero p { color: #888; font-size: 0.9rem; letter-spacing: 0.05em; }
.accent { color: #c8f55a; }
[data-testid="stSidebar"] { background: #111 !important; border-right: 1px solid #222; }
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace !important; }
[data-testid="stFileUploader"] { background: #161616; border: 1px dashed #333; border-radius: 10px; padding: 0.5rem; }
[data-testid="stFileUploader"]:hover { border-color: #c8f55a; }
.stButton > button {
    background: #c8f55a !important; color: #0d0d0d !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.85rem !important; letter-spacing: 0.08em !important;
    border: none !important; border-radius: 6px !important;
    padding: 0.6rem 1.5rem !important; text-transform: uppercase !important;
}
.stButton > button:hover { background: #d9ff6e !important; transform: translateY(-1px); }
.chat-user {
    background: #1a1a1a; border: 1px solid #2a2a2a; border-left: 3px solid #c8f55a;
    border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0; font-size: 0.88rem;
}
.chat-ai {
    background: #161616; border: 1px solid #222; border-left: 3px solid #555;
    border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0;
    font-size: 0.88rem; line-height: 1.7;
}
.label {
    font-family: 'Syne', sans-serif; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.4rem;
}
.label-user { color: #c8f55a; }
.label-ai   { color: #888; }
.badge {
    display: inline-block; background: #1e1e1e; border: 1px solid #333;
    border-radius: 4px; padding: 0.2rem 0.6rem; font-size: 0.72rem;
    color: #aaa; margin: 0.2rem 0.2rem 0.2rem 0;
}
.badge-green { border-color: #c8f55a; color: #c8f55a; }
[data-testid="stTextInput"] input {
    background: #161616 !important; border: 1px solid #2a2a2a !important;
    border-radius: 8px !important; color: #e8e8e8 !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.88rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #c8f55a !important;
    box-shadow: 0 0 0 2px rgba(200,245,90,0.1) !important;
}
hr { border-color: #1e1e1e !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


#  Session State 

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# ── Sidebar 
with st.sidebar:
    st.markdown("### Configuration")
    
    st.markdown("---")
    uploaded_files = st.file_uploader("Drop files here", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    build_btn = st.button("Build Knowledge Base", use_container_width=True)
    
    if build_btn:
       
        if not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            with st.spinner("Processing documents..."):
                try:
                    file_data = [(f.name, f.read()) for f in uploaded_files]
                    docs = load_documents(file_data)
                    vectorstore = create_vectorstore(docs)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                    st.session_state.rag_chain = build_rag_chain(retriever)
                    st.session_state.docs_loaded = [f.name for f in uploaded_files]
                    st.session_state.nchunks = len(docs)
                    st.success(f"Ready! {st.session_state.nchunks} chunks indexed.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    if st.session_state.docs_loaded:
        for name in st.session_state.docs_loaded:
            st.markdown(f'<span class="badge badge-green">{name}</span>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_history()
        st.rerun()

#  Main Area 
st.markdown("""
<div class="hero">
    <h1>📄 RAG Document <span class="accent">Q&A</span></h1>
    <p>Upload your documents → Ask anything → Get grounded answers</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Load & display history
history = load_history()
if history:
    for msg in history:
        if msg["role"] == "user":
            st.markdown(f'''
                <div class="chat-user">
                    <div class="label label-user">You</div>
                    {msg["content"]}
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="chat-ai">
                    <div class="label label-ai">Assistant</div>
                    {msg["content"]}
                </div>
            ''', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:#555; text-align:center; font-size:0.85rem;">Start chatting!</p>', unsafe_allow_html=True)

st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input("Ask a question...", placeholder="What does the document say about...")
with col2:
    ask_btn = st.button("Ask", use_container_width=True)

if ask_btn and question:
    if not st.session_state.rag_chain:
        st.warning("Build knowledge base first!")
    else:
        add_message("user", question)  # Save user Q
        
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.rag_chain.invoke(question)
            except Exception as e:
                answer = f"Error: {e}"
        
        add_message("assistant", answer)  # Save response
        st.rerun()