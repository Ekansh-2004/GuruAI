import os
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain_core.messages import HumanMessage, AIMessage

from src.rag.loader import load_documents
from src.rag.embedder import create_vectorstore, load_existing_vectorstore
from src.rag.chain import build_rag_chain
from src.rag.quiz import generate_quiz_for_session_db
from src.rag.topic_tutor import generate_topic_explanation, generate_topic_quiz
import src.personalization.tracker as tracker
import src.personalization.user_memory as user_memory

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory retriever cache (vectorstore loading is expensive) ──
_retriever_cache: dict = {}

def get_retriever(session_id: str):
    """Cache only the retriever (vectorstore) — the expensive part."""
    if session_id not in _retriever_cache:
        db = load_existing_vectorstore(session_id)
        if db:
            _retriever_cache[session_id] = db.as_retriever(search_kwargs={"k": 4})
        else:
            return None
    return _retriever_cache[session_id]


def build_profile_summary() -> str:
    """
    Serialize the global knowledge profile into a compact text block
    the LLM can read and reason about.
    Format:
      Subject: Deep Learning
        - Weak  (0%): Activation Functions
        - Strong (100%): Neural Networks
    """
    profile = tracker.get_performance_areas()
    if not profile:
        return ""
    lines = []
    for subject, levels in profile.items():
        subject_lines = []
        for level_name, items in levels.items():
            for item in items:
                topic, score = item[0], item[1]
                pct = round(score * 100)
                subject_lines.append(f"    - {level_name.capitalize()} ({pct}%): {topic}")
        if subject_lines:
            lines.append(f"  Subject: {subject}")
            lines.extend(subject_lines)
    return "\n".join(lines) if lines else ""


# ── Serve Frontend ──────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/index.html")
def root_index():
    return FileResponse("static/index.html")

@app.get("/knowledge.html")
def knowledge():
    return FileResponse("static/knowledge.html")

@app.get("/topic.html")
def topic_page():
    return FileResponse("static/topic.html")


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
    _retriever_cache.pop(session_id, None)
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
    _retriever_cache[session_id] = vectorstore.as_retriever(search_kwargs={"k": 4})

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
    retriever = get_retriever(req.session_id)
    if not retriever:
        raise HTTPException(status_code=400, detail="No database built for this session. Please upload documents first.")

    profile_summary = build_profile_summary()
    memory_context = user_memory.get_memory_as_system_context()
    chain = build_rag_chain(retriever, knowledge_profile_summary=profile_summary, user_memory_context=memory_context)

    history_raw = tracker.get_session_messages(req.session_id)
    # Filter out quiz messages from LLM history
    text_history = [m for m in history_raw if m["role"] in ("user", "assistant")]

    if not text_history:
        title = req.question[:25] + "..." if len(req.question) > 25 else req.question
        tracker.update_session_title(req.session_id, title)

    tracker.add_message(req.session_id, "user", req.question)

    formatted_history = []
    for h in text_history:
        if h["role"] == "user":
            formatted_history.append(HumanMessage(content=h["content"]))
        else:
            formatted_history.append(AIMessage(content=h["content"]))

    def generate():
        full_response = ""
        try:
            for chunk in chain.stream({"question": req.question, "chat_history": formatted_history}):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if full_response:
                tracker.add_message(req.session_id, "assistant", full_response)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Quiz ────────────────────────────────────────────────
@app.post("/api/sessions/{session_id}/quiz")
def generate_quiz(session_id: str):
    path = f"faiss_index_db/{session_id}/index.faiss"
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="Build the database first.")
    quiz = generate_quiz_for_session_db(session_id)
    if not quiz or not quiz.get("questions"):
        raise HTTPException(status_code=500, detail="Quiz generation failed.")
    tracker.save_quiz(session_id, quiz)         # Keep for topic.html backwards compat
    tracker.add_quiz_message(session_id, quiz)  # Embed in chat feed for inline rendering
    return quiz

@app.get("/api/sessions/{session_id}/quiz")
def get_quiz(session_id: str):
    return tracker.get_quiz(session_id)

class QuizAnswerRequest(BaseModel):
    session_id: str
    subject: str
    topic: str
    is_correct: bool

@app.post("/api/quiz/answer")
def submit_answer(req: QuizAnswerRequest):
    tracker.update_topic_performance(req.session_id, req.subject, req.topic, req.is_correct)
    return {"status": "recorded"}


# ── Knowledge Profile ───────────────────────────────────
@app.get("/api/profile")
def get_profile():
    return tracker.get_performance_areas()


# ── Topic Tutor ─────────────────────────────────────────
class TopicExplainRequest(BaseModel):
    topic: str
    subject: str
    mastery_level: str  # "strong", "average", "weak"
    score_pct: int      # 0-100

@app.post("/api/topic/explain")
def explain_topic(req: TopicExplainRequest):
    explanation = generate_topic_explanation(
        req.topic, req.subject, req.mastery_level, req.score_pct
    )
    return {"explanation": explanation}

@app.post("/api/topic/quiz")
def topic_quiz(req: TopicExplainRequest):
    quiz = generate_topic_quiz(
        req.topic, req.subject, req.mastery_level, req.score_pct
    )
    return quiz


# ── User Memory ────────────────────────────────────────────

class MemoryMessageRequest(BaseModel):
    message: str

@app.get("/api/memory")
def get_memory():
    """Return all stored memory items."""
    return {"items": user_memory.load_memory()}

@app.post("/api/memory")
def add_memory(req: MemoryMessageRequest):
    """
    Accept a free-form message from the user, extract preferences via LLM,
    store them, and return the full updated memory list.
    """
    extracted = user_memory.extract_preferences_from_message(req.message)
    updated = user_memory.add_memory_items(extracted)
    return {"extracted": extracted, "items": updated}

@app.delete("/api/memory/{index}")
def delete_memory(index: int):
    """Delete a memory item by its index."""
    updated = user_memory.delete_memory_item(index)
    return {"items": updated}

@app.delete("/api/memory")
def clear_memory():
    """Clear all memory items."""
    user_memory.clear_all_memory()
    return {"items": []}

@app.get("/api/memory/chat")
def get_memory_chat():
    """Return the persistent memory chat history."""
    return {"history": user_memory.get_chat_history()}

@app.post("/api/memory/chat")
def memory_chat_message(req: MemoryMessageRequest):
    """
    Send a message to the memory chat bot.
    It responds conversationally and silently extracts + stores preferences.
    """
    reply, extracted = user_memory.memory_chat(req.message)
    return {
        "reply": reply,
        "extracted": extracted,
        "items": user_memory.load_memory(),
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)