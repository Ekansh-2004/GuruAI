import json
import os
import uuid
import shutil
from datetime import datetime
from typing import Dict, List
from src.core.database import get_db

# EMA smoothing factor
_EMA_ALPHA = 0.3

def load_all_sessions(user_id: int) -> Dict:
    """Load the session dictionary for a specific user from SQLite."""
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute(
        "SELECT id, title, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC", 
        (user_id,)
    )
    rows = cur.fetchall()
    
    sessions = {}
    for r in rows:
        sid = r["id"]
        
        # Fetch messages
        cur.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", 
            (sid,)
        )
        messages = []
        for mr in cur.fetchall():
            role = mr["role"]
            content = mr["content"]
            if role == "quiz":
                try:
                    content = json.loads(content)
                except Exception:
                    pass
            messages.append({"role": role, "content": content})
            
        # Fetch documents
        cur.execute(
            "SELECT name, size FROM documents WHERE session_id = ?", 
            (sid,)
        )
        documents = [{"name": dr["name"], "size": dr["size"]} for dr in cur.fetchall()]
        
        sessions[sid] = {
            "title": r["title"],
            "messages": messages,
            "documents": documents,
            "updated_at": r["updated_at"]
        }
        
    return sessions

def create_session(user_id: int, title: str = "New Chat") -> str:
    """Generate a new session ID for a user and save it to SQLite."""
    session_id = str(uuid.uuid4())
    now = str(datetime.now())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, title, now, now)
        )
        conn.commit()
    return session_id

def get_session_messages(session_id: str) -> List[Dict[str, str]]:
    """Get message history for one specific session from SQLite."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", 
        (session_id,)
    )
    messages = []
    for r in cur.fetchall():
        role = r["role"]
        content = r["content"]
        if role == "quiz":
            try:
                content = json.loads(content)
            except Exception:
                pass
        messages.append({"role": role, "content": content})
    return messages

def add_message(session_id: str, role: str, content: str):
    """Append a message to a specific session in SQLite and update timestamp."""
    now = str(datetime.now())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", 
            (session_id, role, content)
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", 
            (now, session_id)
        )
        conn.commit()

def add_quiz_message(session_id: str, quiz: dict):
    """Store a quiz as an inline message in the chat feed (role='quiz') in SQLite, preserving ordering."""
    now = str(datetime.now())
    quiz_str = json.dumps(quiz)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'quiz', ?)", 
            (session_id, quiz_str)
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", 
            (now, session_id)
        )
        conn.commit()

def update_session_title(session_id: str, new_title: str):
    """Rename a session in SQLite."""
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?", 
            (new_title, session_id)
        )
        conn.commit()

def delete_session(session_id: str):
    """Delete a chat session entirely from SQLite, including its isolated FAISS database."""
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        
    # Attempt to remove the FAISS vector database
    db_path = os.path.join(os.getcwd(), "faiss_index_db", session_id)
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except Exception as e:
            print(f"Error removing db folder: {e}")

def add_document(session_id: str, filename: str, size: int):
    """Save the uploaded document's metadata to the session in SQLite."""
    with get_db() as conn:
        cur = conn.cursor()
        # Prevent duplicates
        cur.execute(
            "SELECT id FROM documents WHERE session_id = ? AND name = ?", 
            (session_id, filename)
        )
        if not cur.fetchone():
            conn.execute(
                "INSERT INTO documents (session_id, name, size) VALUES (?, ?, ?)", 
                (session_id, filename, size)
            )
            conn.commit()

def get_session_documents(session_id: str) -> list:
    """Retrieve the list of documents for a specific session from SQLite."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT name, size FROM documents WHERE session_id = ?", 
        (session_id,)
    )
    return [{"name": r["name"], "size": r["size"]} for r in cur.fetchall()]

def clear_session_knowledge_base(session_id: str):
    """Wipe FAISS files and document metadata from SQLite."""
    # 1. Clear the hard drive folder
    db_path = os.path.join(os.getcwd(), "faiss_index_db", session_id)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    # 2. Clear the SQLite rows
    with get_db() as conn:
        conn.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
        conn.commit()

def save_quiz(session_id: str, quiz: dict):
    """Persist the generated quiz inside the session in SQLite so it survives navigation."""
    quiz_str = json.dumps(quiz)
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET quiz_json = ? WHERE id = ?", 
            (quiz_str, session_id)
        )
        conn.commit()

def get_quiz(session_id: str) -> dict:
    """Retrieve the saved quiz for a session from SQLite, or empty if none."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT quiz_json FROM sessions WHERE id = ?", (session_id,))
    row = cur.fetchone()
    if row and row["quiz_json"]:
        try:
            return json.loads(row["quiz_json"])
        except Exception:
            return {}
    return {}

# ── Global Student Profile / Knowledge Base ──

def load_global_profile(user_id: int) -> Dict:
    """Load the globally aggregated knowledge profile for a user from SQLite."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT subject, topic, correct, total, ema_score FROM knowledge_profile WHERE user_id = ?", 
        (user_id,)
    )
    profile = {}
    for r in cur.fetchall():
        subj = r["subject"]
        topic = r["topic"]
        if subj not in profile:
            profile[subj] = {}
        profile[subj][topic] = {
            "correct": r["correct"],
            "total": r["total"],
            "ema_score": r["ema_score"]
        }
    return profile

def update_topic_performance(session_id: str, subject: str, topic: str, correct: bool):
    """Record quiz performance and update the EMA score for the topic globally in SQLite."""
    conn = get_db()
    cur = conn.cursor()
    
    # 1. Fetch user_id from the session
    cur.execute("SELECT user_id FROM sessions WHERE id = ?", (session_id,))
    sess_row = cur.fetchone()
    if not sess_row:
        return  # Session not found
    user_id = sess_row["user_id"]
    
    subj = subject.title().strip()
    t = topic.title().strip()
    
    # 2. Query existing topics under this subject for fuzzy matching
    cur.execute(
        "SELECT topic FROM knowledge_profile WHERE user_id = ? AND subject = ?", 
        (user_id, subj)
    )
    existing_topics = [r["topic"] for r in cur.fetchall()]
    
    import difflib
    matches = difflib.get_close_matches(t, existing_topics, n=1, cutoff=0.85)
    if matches:
        t = matches[0]
        
    # 3. Retrieve stats
    cur.execute(
        "SELECT correct, total, ema_score FROM knowledge_profile WHERE user_id = ? AND subject = ? AND topic = ?",
        (user_id, subj, t)
    )
    row = cur.fetchone()
    if not row:
        correct_val = 0
        total_val = 0
        ema = None
    else:
        correct_val = row["correct"]
        total_val = row["total"]
        ema = row["ema_score"]
        
    total_val += 1
    if correct:
        correct_val += 1
        
    recent = 1.0 if correct else 0.0
    prev = ema if ema is not None else 0.0
    new_ema = (recent * _EMA_ALPHA) + (prev * (1 - _EMA_ALPHA))
    
    # 4. Insert or update entry
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO knowledge_profile (user_id, subject, topic, correct, total, ema_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, subject, topic) DO UPDATE SET
                correct = excluded.correct,
                total = excluded.total,
                ema_score = excluded.ema_score,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, subj, t, correct_val, total_val, new_ema)
        )
        conn.commit()

def get_performance_areas(user_id: int) -> Dict:
    """Returns globally tracked topics for a user classified into weak, average, and strong."""
    profile = load_global_profile(user_id)
    
    result = {}
    for subj, topics in profile.items():
        weak, average, strong = [], [], []
        for t, stats in topics.items():
            if stats["total"] == 0: 
                continue
            score = stats["ema_score"] if stats.get("ema_score") is not None else (stats["correct"] / stats["total"])
            val = (t, score, stats["correct"], stats["total"])
            if score < 0.5:
                weak.append(val)
            elif score <= 0.75:
                average.append(val)
            else:
                strong.append(val)
                
        result[subj] = {
            "weak": sorted(weak, key=lambda x: x[1]),
            "average": sorted(average, key=lambda x: x[1]),
            "strong": sorted(strong, key=lambda x: x[1], reverse=True)
        }
            
    return result

def delete_topic(user_id: int, subject: str, topic: str):
    """Remove a topic's score data from the global knowledge profile in SQLite."""
    subj_key = subject.title().strip()
    topic_key = topic.title().strip()
    with get_db() as conn:
        conn.execute(
            "DELETE FROM knowledge_profile WHERE user_id = ? AND subject = ? AND topic = ?",
            (user_id, subj_key, topic_key)
        )
        conn.commit()