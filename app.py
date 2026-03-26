import streamlit as st
import os
from src.rag.loader import load_documents
from src.rag.embedder import create_vectorstore, load_existing_vectorstore
from src.rag.chain import build_rag_chain

# Convert simple dict history to LangChain message objects
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Study AI", page_icon="📚", layout="wide")

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Startup: Try to load existing database ---
if st.session_state.rag_chain is None:
    existing_db = load_existing_vectorstore()
    if existing_db:
        retriever = existing_db.as_retriever(search_kwargs={"k": 4})
        st.session_state.rag_chain = build_rag_chain(retriever)

# --- Sidebar ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    
    if st.button("Build Database"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                file_data = [(f.name, f.read()) for f in uploaded_files]
                docs = load_documents(file_data)
                vectorstore = create_vectorstore(docs) # This now saves to disk!
                
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                st.session_state.rag_chain = build_rag_chain(retriever)
                st.success("Database built and saved!")

    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.rerun()

# --- Main UI ---
st.title("Study AI Tutor 🤖")

if st.session_state.rag_chain is None:
    st.warning("Please upload documents and build the database to start.")

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if question := st.chat_input("Ask a question..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.history.append({"role": "user", "content": question})
    
    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # Format history for LangChain
                formatted_history = []
                for h in st.session_state.history[:-1]: # Exclude the current question
                    if h["role"] == "user":
                        formatted_history.append(HumanMessage(content=h["content"]))
                    else:
                        formatted_history.append(AIMessage(content=h["content"]))
                
                # Invoke chain with question AND history
                answer = st.session_state.rag_chain.invoke({
                    "question": question,
                    "chat_history": formatted_history
                })
                
                st.markdown(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})