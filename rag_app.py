import os
import tempfile
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page Config 
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
)
#AIzaSyDeIRczVdy2PqaixPeu7pDTbz-Nh0pPcRA
#Custom CSS 
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

#Helpers
def load_document(file_path: str, original_name: str):
    ext = os.path.splitext(original_name)[1].lower()
    loaders = {".pdf": PyPDFLoader, ".txt": TextLoader, ".docx": Docx2txtLoader}
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    return loaders[ext](file_path).load()

@st.cache_resource(show_spinner=False)
def build_rag_chain_cached(file_data_tuple, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
   
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    all_docs = []
    for name, data in file_data_tuple:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        all_docs.extend(load_document(tmp_path, name))
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
You are a precise assistant. Answer the question with the help of the context below and your own knowledge.
If the answer is not found in the context. Answer it with your understanding"

Context:
{context}

Question: {question}

Answer:
""")

    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain, len(chunks)

# ── Sidebar 
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.text_input(
        "Google API Key",
        type="password",
        placeholder="Enter your Google API Key...",
        key="api_key", 
    )
    st.markdown("---")

    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    build_btn = st.button("🔨 Build Knowledge Base", use_container_width=True)

    if build_btn:
        current_key = st.session_state.api_key
        if not current_key:
            st.error("Please enter your Google API Key.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            with st.spinner("Processing documents..."):
                try:
                    file_data = tuple((f.name, f.read()) for f in uploaded_files)
                    chain, n_chunks = build_rag_chain_cached(file_data, current_key)
                    st.session_state.rag_chain = chain
                    st.session_state.docs_loaded = [f.name for f in uploaded_files]
                    st.session_state.chat_history = []
                    st.success(f"Ready! {n_chunks} chunks indexed.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    if st.session_state.docs_loaded:
        st.markdown("**Indexed files:**")
        for name in st.session_state.docs_loaded:
            st.markdown(f'<span class="badge badge-green">✓ {name}</span>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

#  Main Area 
st.markdown("""
<div class="hero">
    <h1>📄 RAG Document <span class="accent">Q&A</span></h1>
    <p>Upload your documents → Ask anything → Get grounded answers</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

if not st.session_state.chat_history:
    if st.session_state.rag_chain:
        st.markdown('<p style="color:#555; text-align:center; font-size:0.85rem;">Knowledge base ready. Ask your first question below.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#555; text-align:center; font-size:0.85rem;">Upload documents in the sidebar to get started.</p>', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-user">
            <div class="label label-user">You</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-ai">
            <div class="label label-ai">Assistant</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)

st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Ask a question",
        placeholder="What does the document say about...?",
        label_visibility="collapsed",
        key="question_input",
    )
with col2:
    ask_btn = st.button("Ask →", use_container_width=True)

if ask_btn and question:
    if not st.session_state.rag_chain:
        st.warning("Please build the knowledge base first using the sidebar.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.rag_chain.invoke(question)
            except Exception as e:
                answer = f"Error generating answer: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()