import os
import json
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, status, Query
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage

from src.rag.loader import load_documents
from src.rag.embedder import create_vectorstore, load_existing_vectorstore, get_db_path
from src.rag.chain import build_rag_chain
from src.rag.quiz import generate_quiz_for_session_db
from src.rag.topic_tutor import generate_topic_explanation, generate_topic_quiz
from langchain_community.retrievers import TFIDFRetriever
from src.rag.retriever import HybridRetriever
import src.personalization.tracker as tracker
import src.personalization.user_memory as user_memory

from src.auth.auth import hash_password, verify_password, create_access_token, verify_access_token
from src.core.database import get_db, init_db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SQLite database on startup
@app.on_event("startup")
def startup_event():
    init_db()

# ── In-memory retriever cache (LRU-bounded) ──
from collections import OrderedDict
_retriever_cache: OrderedDict = OrderedDict()
_RETRIEVER_CACHE_MAX = 32  # Each entry holds a FAISS index (~10-50MB in RAM)

def get_retriever(session_id: str):
    """Return a cached retriever, loading from disk if needed. LRU-evicts oldest when full."""
    if session_id in _retriever_cache:
        _retriever_cache.move_to_end(session_id)  # Mark as recently used
        return _retriever_cache[session_id]
    db = load_existing_vectorstore(session_id)
    if not db:
        return None
    
    # Extract documents from FAISS to build the sparse TF-IDF retriever dynamically
    docs = list(db.docstore._dict.values())
    tfidf_retriever = TFIDFRetriever.from_documents(docs)
    
    retriever = HybridRetriever(vectorstore=db, tfidf_retriever=tfidf_retriever, top_n=4)
    _retriever_cache[session_id] = retriever
    if len(_retriever_cache) > _RETRIEVER_CACHE_MAX:
        _retriever_cache.popitem(last=False)  # Evict least-recently-used
    return retriever

# ── Authentication Dependencies ──

def get_current_user(request: Request) -> int:
    """Dependency to retrieve the currently logged in user ID from the HTTPOnly cookie."""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Session expired or not authenticated"
        )
    payload = verify_access_token(token)
    if not payload or "user_id" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid authentication token"
        )
    return payload["user_id"]

def check_auth_html(request: Request) -> Optional[int]:
    """Helper to check credentials silently for page serving (redirecting instead of JSON error)."""
    token = request.cookies.get("access_token")
    if token:
        payload = verify_access_token(token)
        if payload and "user_id" in payload:
            return payload["user_id"]
    return None

def verify_session_ownership(session_id: str, user_id: int):
    """Raises 403/404 if the session does not belong to the authenticated user."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        if row["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

def build_profile_summary(user_id: int) -> str:
    """
    Serialize the global knowledge profile into a compact text block
    the LLM can read and reason about.
    """
    profile = tracker.get_performance_areas(user_id)
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

# ── Authentication Endpoints ──

class RegisterRequest(BaseModel):
    username: str
    password: str
    name: Optional[str] = "The Scholar"

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/auth/register")
def register(req: RegisterRequest, response: Response):
    username = req.username.strip().lower()
    if not username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password are required")
        
    # Open the database JUST ONCE for the entire function
    with get_db() as conn:
        cur = conn.cursor()
        
        # 1. Check if username exists using  single connection
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Username is already taken")
            
        # 2. Hash password and write the new user using the SAME connection
        pwd_hash = hash_password(req.password)
        cur.execute(
            "INSERT INTO users (username, password_hash, name) VALUES (?, ?, ?)",
            (username, pwd_hash, req.name or "The Scholar")
        )
        conn.commit()
        user_id = cur.lastrowid
        
    # 3. Create the wristband token and set the browser cookie
    token = create_access_token({"user_id": user_id})
    response.set_cookie(
        key="access_token", 
        value=token, 
        httponly=True, 
        max_age=86400, 
        samesite="lax",
        secure=False  
    )
    return {"status": "ok", "user_id": user_id}

@app.post("/api/auth/login")
def login(req: LoginRequest, response: Response):
    username = req.username.strip().lower()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()

    if not row or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")
        
    user_id = row["id"]
    token = create_access_token({"user_id": user_id})
    response.set_cookie(
        key="access_token", 
        value=token, 
        httponly=True, 
        max_age=86400, 
        samesite="lax",
        secure=False
    )
    return {"status": "ok", "user_id": user_id}

@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    return {"status": "ok"}

# ── Serve Frontend ──
"""
Every single one of these routes is just checking if a user has permission to view a layout file before letting them see it. It keeps strangers out of the main application and stops logged-in users from seeing the login screen.
"""
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root(request: Request):
    user_id = check_auth_html(request)
    if not user_id:
        return RedirectResponse(url="/login.html")
    return FileResponse("static/index.html")

@app.get("/index.html")
def root_index(request: Request):
    user_id = check_auth_html(request)
    if not user_id:
        return RedirectResponse(url="/login.html")
    return FileResponse("static/index.html")

@app.get("/knowledge.html")
def knowledge(request: Request):
    user_id = check_auth_html(request)
    if not user_id:
        return RedirectResponse(url="/login.html")
    return FileResponse("static/knowledge.html")

@app.get("/topic.html")
def topic_page(request: Request):
    user_id = check_auth_html(request)
    if not user_id:
        return RedirectResponse(url="/login.html")
    return FileResponse("static/topic.html")

@app.get("/user.html")
def user_page(request: Request):
    user_id = check_auth_html(request)
    if not user_id:
        return RedirectResponse(url="/login.html")
    return FileResponse("static/user.html")

@app.get("/login.html")
def login_page(request: Request):
    user_id = check_auth_html(request)
    if user_id:
        return RedirectResponse(url="/index.html")
    return FileResponse("static/login.html")

# ── Sessions ──
"""
Unlike the previous routes—which serve full visual HTML pages—these endpoints only send and receive raw data packets (JSON text). They all use Depends(get_current_user) as a locked gate, meaning a browser can only call them if the user has a valid, logged-in wristband.
"""
@app.get("/api/sessions")
def get_sessions(user_id: int = Depends(get_current_user)):
    return tracker.load_all_sessions(user_id)

@app.post("/api/sessions")
def create_session(user_id: int = Depends(get_current_user)):
    session_id = tracker.create_session(user_id, "New Chat")
    return {"session_id": session_id}

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user_id: int = Depends(get_current_user)):
    verify_session_ownership(session_id, user_id)
    tracker.delete_session(session_id)
    _retriever_cache.pop(session_id, None)
    return {"status": "deleted"}

@app.patch("/api/sessions/{session_id}/title")
def update_title(
    session_id: str, 
    title: str = Form(...), 
    user_id: int = Depends(get_current_user)
):
    verify_session_ownership(session_id, user_id)
    tracker.update_session_title(session_id, title)
    return {"status": "updated"}

@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str, user_id: int = Depends(get_current_user)):
    verify_session_ownership(session_id, user_id)
    return tracker.get_session_messages(session_id)

@app.get("/api/sessions/{session_id}/db-status")
def db_status(session_id: str, user_id: int = Depends(get_current_user)):
    verify_session_ownership(session_id, user_id)
    path = f"faiss_index_db/{session_id}/index.faiss"
    return {"exists": os.path.exists(path)}

# ── Upload & Build Vectorstore ──
@app.post("/api/sessions/{session_id}/upload")
async def upload_and_build(
    session_id: str,
    files: List[UploadFile] = File(...),
    user_id: int = Depends(get_current_user)
):
    """Upload one or more PDF/DOCX/TXT files into a session's shared knowledge base.

    Each file is tracked as its own document (id, type, status, storage location)
    and every resulting chunk is tagged with the document it came from (and page
    number, for paginated file types) so retrieved content stays traceable to its
    source. A file that fails to parse doesn't block the others in the same batch.
    """
    verify_session_ownership(session_id, user_id)

    all_docs = []
    results = []
    for f in files:
        content = await f.read()
        doc_id = str(uuid.uuid4())
        file_type = os.path.splitext(f.filename)[1].lower().lstrip(".")
        try:
            chunks = load_documents([(f.filename, content, doc_id)])
            if not chunks:
                raise ValueError("No readable text extracted from this file")

            tracker.add_document(
                session_id, doc_id, f.filename, len(content), file_type,
                status="ready", storage_path=get_db_path(session_id), chunk_count=len(chunks),
            )
            all_docs.extend(chunks)
            results.append({
                "doc_id": doc_id, "filename": f.filename, "file_type": file_type,
                "status": "ready", "chunk_count": len(chunks),
            })
        except Exception as e:
            tracker.add_document(
                session_id, doc_id, f.filename, len(content), file_type,
                status="failed", storage_path=None, chunk_count=0, error=str(e),
            )
            results.append({
                "doc_id": doc_id, "filename": f.filename, "file_type": file_type,
                "status": "failed", "error": str(e),
            })

    if not all_docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="None of the uploaded file(s) contained readable text. Please ensure they are not empty or scanned images without OCR."
        )

    # Merge the new chunks into the session's vectorstore (existing docs, if any, are kept)
    vectorstore = create_vectorstore(all_docs, session_id) #transforms all the paragraphs into mathematical coordinates

    # Rebuild the hybrid retriever over the FULL session docstore, not just this batch
    full_docs = list(vectorstore.docstore._dict.values())
    tfidf_retriever = TFIDFRetriever.from_documents(full_docs)
    _retriever_cache[session_id] = HybridRetriever(vectorstore=vectorstore, tfidf_retriever=tfidf_retriever, top_n=4)

    return {"status": "database built", "doc_count": len(all_docs), "documents": results}

@app.get("/api/sessions/{session_id}/documents")
def get_documents(session_id: str, user_id: int = Depends(get_current_user)):
    verify_session_ownership(session_id, user_id)
    return tracker.get_session_documents(session_id)

@app.delete("/api/sessions/{session_id}/knowledge")
def delete_knowledge_base(session_id: str, user_id: int = Depends(get_current_user)):
    """Wipe the FAISS vector store and document list for a session, keeping chat history intact."""
    verify_session_ownership(session_id, user_id)
    tracker.clear_session_knowledge_base(session_id)
    _retriever_cache.pop(session_id, None)
    return {"status":"knowledge base cleared"}

# ── Chat ──
class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.post("/api/chat")
def chat(req: ChatRequest, user_id: int = Depends(get_current_user)):
    verify_session_ownership(req.session_id, user_id)
    retriever = get_retriever(req.session_id)
    if not retriever:
        raise HTTPException(
            status_code=400, 
            detail="No database built for this session. Please upload documents first."
        )

    profile_summary = build_profile_summary(user_id)
    print(profile_summary)
    memory_context = user_memory.get_memory_as_system_context(user_id)
    chain = build_rag_chain(retriever, knowledge_profile_summary=profile_summary, user_memory_context=memory_context)

    history_raw = tracker.get_session_messages(req.session_id)
    # Filter out quiz messages from LLM history
    text_history = [m for m in history_raw if m["role"] in ("user", "assistant")]

    # Token optimization: Limit history to the last 4 messages (2 QA pairs)
    if len(text_history) > 4:
        text_history = text_history[-4:]

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

    # ── Run CRAG before streaming so we capture sources metadata ────────────
    from src.rag.crag import build_crag_context
    context_text, source_label, sources_metadata = build_crag_context(retriever, req.question)
    print(f"[Chat] Source label: {source_label} | Sources count: {len(sources_metadata)}")

    def generate():
        full_response = ""
        try:
            for chunk in chain.stream({
                "question": req.question,
                "context": context_text,
                "chat_history": formatted_history,
            }):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            # Emit sources metadata before [DONE] so the UI can attach the drawer
            yield f"data: {json.dumps({'sources': sources_metadata})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if full_response:
                tracker.add_message(req.session_id, "assistant", full_response)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# ── Quiz ──
@app.post("/api/sessions/{session_id}/quiz")
def generate_quiz(session_id: str, user_id: int = Depends(get_current_user)):
    verify_session_ownership(session_id, user_id)
    path = f"faiss_index_db/{session_id}/index.faiss"
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="Build the database first.")
    quiz = generate_quiz_for_session_db(session_id, user_subjects=user_memory.load_subjects(user_id))
    if not quiz or not quiz.get("questions"):
        raise HTTPException(status_code=500, detail="Quiz generation failed.")
    tracker.save_quiz(session_id, quiz)         # Keep for topic.html backwards compat
    tracker.add_message(session_id, "quiz", quiz)  # Embed in chat feed for inline rendering
    return quiz

@app.get("/api/sessions/{session_id}/quiz")
def get_quiz(session_id: str, user_id: int = Depends(get_current_user)):
    verify_session_ownership(session_id, user_id)
    return tracker.get_quiz(session_id)

class QuizAnswerRequest(BaseModel):
    session_id: str
    subject: str
    topic: str
    is_correct: bool

@app.post("/api/quiz/answer")
def submit_answer(req: QuizAnswerRequest, user_id: int = Depends(get_current_user)):
    verify_session_ownership(req.session_id, user_id)
    tracker.update_topic_performance(req.session_id, req.subject, req.topic, req.is_correct)
    return {"status": "recorded"}

# ── Knowledge Profile ──
@app.get("/api/profile")
def get_profile(user_id: int = Depends(get_current_user)):
    return tracker.get_performance_areas(user_id)

@app.delete("/api/profile/{subject}/{topic}")
def delete_profile_topic(subject: str, topic: str, user_id: int = Depends(get_current_user)):
    """Permanently remove a topic from the student's global knowledge profile."""
    tracker.delete_topic(user_id, subject, topic)
    return {"status": "deleted", "subject": subject, "topic": topic}

# ── Spaced Repetition (SRS) ──
@app.get("/api/suggestions/review-queue")
def get_review_queue(
    category: str = Query("all", pattern="^(all|weak|average|strong)$"),
    limit: int = Query(10, ge=1, le=50),
    sort: str = Query("urgent", pattern="^(urgent|recent)$"),
    user_id: int = Depends(get_current_user),
) -> dict:
    """Return topics due for review today or earlier.

    category filters by mastery_category ('weak'/'average'/'strong'), sort picks
    between soonest-due-first ('urgent') and most-recently-studied-first ('recent').
    """
    topics = tracker.list_topics_with_schedule(user_id)
    due_topics = [t for t in topics if t["is_due"]]

    if category != "all":
        due_topics = [t for t in due_topics if t["mastery_category"].lower() == category]

    if sort == "urgent":
        due_topics.sort(key=lambda t: t["days_until_review"])
    else:
        due_topics.sort(key=lambda t: t["last_reviewed"] or "", reverse=True)

    overdue_count = sum(1 for t in topics if t["days_until_review"] < 0)

    queue = [
        {
            "id": t["id"],
            "topic": t["topic"],
            "mastery_level": t["mastery_level"],
            "mastery_category": t["mastery_category"],
            "days_until_review": t["days_until_review"],
            "last_reviewed": t["last_reviewed"],
            "next_review": t["next_review"],
            "review_count": t["review_count"],
            "urgency_score": t["urgency_score"],
        }
        for t in due_topics[:limit]
    ]

    return {
        "queue": queue,
        "total_topics": len(topics),
        "overdue_count": overdue_count,
    }

class MarkReviewedRequest(BaseModel):
    score: int
    notes: Optional[str] = None

@app.post("/api/topics/{topic_id}/mark-reviewed")
def mark_topic_reviewed(
    topic_id: int,
    req: MarkReviewedRequest,
    user_id: int = Depends(get_current_user),
) -> dict:
    """Record a study session for a topic: updates its mastery EMA and spaced-repetition schedule.

    score is 0-10. notes is accepted for future use but is not currently persisted
    (knowledge_profile has no notes column).
    """
    if not (0 <= req.score <= 10):
        raise HTTPException(status_code=400, detail="score must be between 0 and 10")

    updated = tracker.update_ema(topic_id, user_id, req.score / 10)
    if not updated:
        raise HTTPException(status_code=404, detail="Topic not found")

    return {
        "topic": updated["topic"],
        "mastery_updated": updated["mastery_level"],
        "next_review": updated["next_review"],
        "message": "Great! Study session recorded.",
    }

@app.get("/api/topics/statistics")
def get_topics_statistics(user_id: int = Depends(get_current_user)) -> dict:
    """Return dashboard stats summarizing the user's spaced-repetition progress."""
    return tracker.get_topic_statistics(user_id)

# ── User Subjects ──
class SubjectRequest(BaseModel):
    subject: str

@app.get("/api/subjects")
def get_subjects(user_id: int = Depends(get_current_user)):
    """Return all registered subjects."""
    return {"subjects": user_memory.load_subjects(user_id)}

@app.post("/api/subjects")
def add_subject(req: SubjectRequest, user_id: int = Depends(get_current_user)):
    """Add a new subject to the user's study profile."""
    updated = user_memory.save_subject(user_id, req.subject)
    return {"subjects": updated}

@app.delete("/api/subjects/{subject}")
def remove_subject(subject: str, user_id: int = Depends(get_current_user)):
    """Remove a subject from the user's study profile."""
    updated = user_memory.delete_subject(user_id, subject)
    return {"subjects": updated}

# ── Topic Tutor ──
class TopicExplainRequest(BaseModel):
    topic: str
    subject: str
    mastery_level: str  # "strong", "average", "weak"
    score_pct: int      # 0-100

@app.post("/api/topic/explain")
def explain_topic(req: TopicExplainRequest, user_id: int = Depends(get_current_user)):
    explanation = generate_topic_explanation(
        req.topic, req.subject, req.mastery_level, req.score_pct
    )
    return {"explanation": explanation}

@app.post("/api/topic/quiz")
def topic_quiz(req: TopicExplainRequest, user_id: int = Depends(get_current_user)):
    quiz = generate_topic_quiz(
        req.topic, req.subject, req.mastery_level, req.score_pct
    )
    return quiz

# ── User Memory ──
class MemoryMessageRequest(BaseModel):
    message: str

@app.get("/api/memory")
def get_memory(user_id: int = Depends(get_current_user)):
    """Return all stored memory items."""
    return {"items": user_memory.load_memory(user_id)}

@app.post("/api/memory")
def add_memory(req: MemoryMessageRequest, user_id: int = Depends(get_current_user)):
    """
    Accept a free-form message from the user, extract preferences via LLM,
    store them, and return the full updated memory list.
    """
    extracted = user_memory.extract_preferences_from_message(user_id, req.message)
    updated = user_memory.add_memory_items(user_id, extracted)
    return {"extracted": extracted, "items": updated}

@app.delete("/api/memory/{index}")
def delete_memory(index: int, user_id: int = Depends(get_current_user)):
    """Delete a memory item by its index."""
    updated = user_memory.delete_memory_item(user_id, index)
    return {"items": updated}

@app.delete("/api/memory")
def clear_memory(user_id: int = Depends(get_current_user)):
    """Clear all memory items."""
    user_memory.clear_all_memory(user_id)
    return {"items": []}

@app.get("/api/memory/chat")
def get_memory_chat(user_id: int = Depends(get_current_user)):
    """Return the persistent memory chat history."""
    return {"history": user_memory.get_chat_history(user_id)}

@app.post("/api/memory/chat")
def memory_chat_message(req: MemoryMessageRequest, user_id: int = Depends(get_current_user)):
    """
    Send a message to the memory chat bot.
    It responds conversationally and silently extracts + stores preferences.
    """
    reply, extracted = user_memory.memory_chat(user_id, req.message)
    return {
        "reply": reply,
        "extracted": extracted,
        "items": user_memory.load_memory(user_id),
    }

# ── User Profile ──
class UserProfileRequest(BaseModel):
    name: str
    bio: str

@app.get("/api/user/profile")
def get_user_profile(user_id: int = Depends(get_current_user)):
    """Return the user's stored display name and bio."""
    return user_memory.load_user_profile(user_id)

@app.post("/api/user/profile")
def save_user_profile(req: UserProfileRequest, user_id: int = Depends(get_current_user)):
    """Persist the user's display name and bio."""
    return user_memory.save_user_profile(user_id, req.name, req.bio)

# ── User Stats ──
@app.get("/api/user/stats")
def get_user_stats(user_id: int = Depends(get_current_user)):
    """
    Compute:
    - total_questions: total number of quiz questions answered across all sessions.
    - average_mastery_pct: average EMA score (0-100) across all unique topics.
    """
    global_profile = tracker.load_global_profile(user_id)
    total_questions = 0
    all_ema_scores = []

    for subj_topics in global_profile.values():
        for stats in subj_topics.values():
            total_questions += stats.get("total", 0)
            score = stats.get("ema_score")
            if score is not None:
                all_ema_scores.append(score)

    avg_mastery = round((sum(all_ema_scores) / len(all_ema_scores)) * 100, 1) if all_ema_scores else 0.0

    return {
        "total_questions": total_questions,
        "average_mastery_pct": avg_mastery,
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)