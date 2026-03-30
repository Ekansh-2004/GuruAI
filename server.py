import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain_core.messages import HumanMessage, AIMessage

from src.rag.loader import load_documents
from src.rag.embedder import create_vectorstore, load_existing_vectorstore
from src.rag.chain import build_rag_chain
from src.rag.quiz import generate_quiz_for_session_db
import src.personalization.tracker as tracker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory chain cache (replaces st.cache_resource) ──
_chain_cache: dict = {}

def get_chain(session_id: str):
    if session_id not in _chain_cache:
        db = load_existing_vectorstore(session_id)
        if db:
            retriever = db.as_retriever(search_kwargs={"k": 4})
            _chain_cache[session_id] = build_rag_chain(retriever)
        else:
            return None
    return _chain_cache[session_id]


# ── Serve Frontend ──────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")


# ── Sessions ────────────────────────────────────────────
@app.get("/api/sessions")
def get_sessions():
    return tracker.load_all_sessions()

@app.post("/api/sessions")
def create_session():
    session_id = tracker.create_session("New Chat")
    return {"session_id": session_id}

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    tracker.delete_session(session_id)
    _chain_cache.pop(session_id, None)
    return {"status": "deleted"}

@app.patch("/api/sessions/{session_id}/title")
def update_title(session_id: str, title: str = Form(...)):
    tracker.update_session_title(session_id, title)
    return {"status": "updated"}

@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str):
    return tracker.get_session_messages(session_id)

@app.get("/api/sessions/{session_id}/db-status")
def db_status(session_id: str):
    path = f"faiss_index_db/{session_id}/index.faiss"
    return {"exists": os.path.exists(path)}


# ── Upload & Build Vectorstore ──────────────────────────
@app.post("/api/sessions/{session_id}/upload")
async def upload_and_build(
    session_id: str,
    files: List[UploadFile] = File(...)
):
    file_data = []
    for f in files:
        content = await f.read()
        file_data.append((f.filename, content))
        
        # <-- ADD THIS LINE to track the file for the UI
        tracker.add_document(session_id, f.filename, len(content)) 

    docs = load_documents(file_data)
    vectorstore = create_vectorstore(docs, session_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    _chain_cache[session_id] = build_rag_chain(retriever)

    return {"status": "database built", "doc_count": len(docs)}

@app.get("/api/sessions/{session_id}/documents")
def get_documents(session_id: str):
    return tracker.get_session_documents(session_id)


# ── Chat ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.post("/api/chat")
def chat(req: ChatRequest):
    chain = get_chain(req.session_id)
    if not chain:
        raise HTTPException(status_code=400, detail="No database built for this session. Please upload documents first.")

    history_raw = tracker.get_session_messages(req.session_id)

    # Auto-rename on first message
    if not history_raw:
        title = req.question[:25] + "..." if len(req.question) > 25 else req.question
        tracker.update_session_title(req.session_id, title)

    tracker.add_message(req.session_id, "user", req.question)

    formatted_history = []
    for h in history_raw:
        if h["role"] == "user":
            formatted_history.append(HumanMessage(content=h["content"]))
        else:
            formatted_history.append(AIMessage(content=h["content"]))

    answer = chain.invoke({
        "question": req.question,
        "chat_history": formatted_history
    })

    tracker.add_message(req.session_id, "assistant", answer)
    return {"answer": answer}


# ── Quiz ────────────────────────────────────────────────
@app.post("/api/sessions/{session_id}/quiz")
def generate_quiz(session_id: str):
    path = f"faiss_index_db/{session_id}/index.faiss"
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="Build the database first.")
    quiz = generate_quiz_for_session_db(session_id)
    if not quiz or not quiz.get("questions"):
        raise HTTPException(status_code=500, detail="Quiz generation failed.")
    return quiz

class QuizAnswerRequest(BaseModel):
    session_id: str
    topic: str
    is_correct: bool

@app.post("/api/quiz/answer")
def submit_answer(req: QuizAnswerRequest):
    tracker.update_topic_performance(req.session_id, req.topic, req.is_correct)
    return {"status": "recorded"}


# ── Knowledge Profile ───────────────────────────────────
@app.get("/api/profile")
def get_profile():
    return tracker.get_performance_areas()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)