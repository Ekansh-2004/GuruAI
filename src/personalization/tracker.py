import json
import os
import uuid
import shutil
from datetime import datetime
from typing import Dict, List

HISTORY_FILE = "chat_history.json"

def load_all_sessions() -> Dict:
    """Load the entire session dictionary from JSON."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                else:
                    return {}
        except:
            return {}
    return {}

def save_all_sessions(data: Dict):
    """Save the session dictionary to JSON."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def create_session(title="New Chat") -> str:
    """Generate a new session ID and save it."""
    data = load_all_sessions()
    session_id = str(uuid.uuid4())
    data[session_id] = {
        "title": title,
        "messages": [],
        "topic_scores": {}, 
        "documents": [], 
        "updated_at": str(datetime.now())
    }
    save_all_sessions(data)
    return session_id

def get_session_messages(session_id: str) -> List[Dict[str, str]]:
    """Get history for one specific session."""
    data = load_all_sessions()
    return data.get(session_id, {}).get("messages", [])

def add_message(session_id: str, role: str, content: str):
    """Append a message to a specific session."""
    data = load_all_sessions()
    if session_id in data:
        data[session_id]["messages"].append({"role": role, "content": content})
        data[session_id]["updated_at"] = str(datetime.now())
        save_all_sessions(data)

def add_quiz_message(session_id: str, quiz: dict):
    """Store a quiz as an inline message in the chat feed (role='quiz'), preserving ordering."""
    data = load_all_sessions()
    if session_id in data:
        data[session_id]["messages"].append({"role": "quiz", "content": quiz})
        data[session_id]["updated_at"] = str(datetime.now())
        save_all_sessions(data)

def update_session_title(session_id: str, new_title: str):
    """Rename a session (useful for auto-naming based on the first prompt)."""
    data = load_all_sessions()
    if session_id in data:
        data[session_id]["title"] = new_title
        save_all_sessions(data)

def delete_session(session_id: str):
    """Delete a chat session entirely, including its isolated FAISS database and knowledge profile contributions."""
    data = load_all_sessions()
    if session_id in data:
        del data[session_id]
        save_all_sessions(data)
        
    # Attempt to remove the FAISS vector database
    db_path = os.path.join(os.getcwd(), "faiss_index_db", session_id)
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except Exception as e:
            print(f"Error removing db folder: {e}")

def add_document(session_id: str, filename: str, size: int):
    """Save the uploaded document's metadata to the session."""
    data = load_all_sessions()
    if session_id in data:
        if "documents" not in data[session_id]:
            data[session_id]["documents"] = []
            
        # Prevent duplicate entries in the UI if the same file is re-uploaded
        if not any(d["name"] == filename for d in data[session_id]["documents"]):
            data[session_id]["documents"].append({"name": filename, "size": size})
        save_all_sessions(data)

def get_session_documents(session_id: str) -> list:
    """Retrieve the list of documents for a specific session."""
    data = load_all_sessions()
    return data.get(session_id, {}).get("documents", [])

def save_quiz(session_id: str, quiz: dict):
    """Persist the generated quiz inside the session so it survives navigation."""
    data = load_all_sessions()
    if session_id in data:
        data[session_id]["quiz"] = quiz
        save_all_sessions(data)

def get_quiz(session_id: str) -> dict:
    """Retrieve the saved quiz for a session, or empty if none."""
    data = load_all_sessions()
    return data.get(session_id, {}).get("quiz", {})

# --- Global Student Profile / Knowledge Base ---

# EMA smoothing factor — controls how much weight the latest answer carries.
# α=0.3 means: new_score = (current_answer × 0.3) + (old_score × 0.7)
# A single correct answer from 0% only raises score to 30%, not 100%.
_EMA_ALPHA = 0.3

PROFILE_FILE = "knowledge_profile.json"

def load_global_profile() -> Dict:
    """Load the globally aggregated knowledge profile."""
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    
    # Run a one-time migration if profile doesn't exist but history does
    data = load_all_sessions()
    global_subjects = {}
    for sid, session in data.items():
        session_scores = session.get("topic_scores", {})
        for subj, topics in session_scores.items():
            s_key = subj.title().strip()
            if s_key not in global_subjects:
                global_subjects[s_key] = {}
            for t, stats in topics.items():
                t_key = t.title().strip()
                if t_key not in global_subjects[s_key]:
                    global_subjects[s_key][t_key] = {"correct": 0, "total": 0, "ema_score": None}
                global_subjects[s_key][t_key]["correct"] += stats.get("correct", 0)
                global_subjects[s_key][t_key]["total"] += stats.get("total", 0)
                session_ema = stats.get("ema_score")
                if session_ema is not None:
                    g = global_subjects[s_key][t_key]["ema_score"]
                    if g is None:
                        global_subjects[s_key][t_key]["ema_score"] = session_ema
                    else:
                        global_subjects[s_key][t_key]["ema_score"] = (session_ema * _EMA_ALPHA) + (g * (1 - _EMA_ALPHA))
    
    save_global_profile(global_subjects)
    return global_subjects

def save_global_profile(data: Dict):
    with open(PROFILE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_topic_performance(session_id: str, subject: str, topic: str, correct: bool):
    """Record quiz performance and update the EMA score for the topic globally."""
    data = load_global_profile()
        
    subj = subject.title().strip()
    if subj not in data:
        data[subj] = {}
        
    t = topic.title().strip()
    
    import difflib
    existing_topics = list(data[subj].keys())
    matches = difflib.get_close_matches(t, existing_topics, n=1, cutoff=0.85)
    if matches:
        t = matches[0]

    if t not in data[subj]:
        # First attempt — no prior history, seed EMA with the first result directly
        data[subj][t] = {"correct": 0, "total": 0, "ema_score": None}
        
    entry = data[subj][t]
    entry["total"] += 1
    if correct:
        entry["correct"] += 1

    recent = 1.0 if correct else 0.0
    prev = entry["ema_score"] if entry["ema_score"] is not None else 0.0
    # EMA formula: NewScore = (recent × α) + (previous × (1 − α))
    entry["ema_score"] = (recent * _EMA_ALPHA) + (prev * (1 - _EMA_ALPHA))
        
    save_global_profile(data)

def get_performance_areas():
    """Returns the globally tracked topics classified into weak, average, and strong."""
    global_subjects = load_global_profile()
    
    result = {}
    for subj, topics in global_subjects.items():
        weak, average, strong = [], [], []
        for t, stats in topics.items():
            if stats["total"] == 0: continue
            # Use EMA score for classification; fall back to raw % for old data without ema_score
            score = stats["ema_score"] if stats.get("ema_score") is not None else (stats["correct"] / stats["total"])
            if score < 0.5:
                weak.append((t, score, stats["correct"], stats["total"]))
            elif score <= 0.75:
                average.append((t, score, stats["correct"], stats["total"]))
            else:
                strong.append((t, score, stats["correct"], stats["total"]))
                
        result[subj] = {
            "weak": sorted(weak, key=lambda x: x[1]),
            "average": sorted(average, key=lambda x: x[1]),
            "strong": sorted(strong, key=lambda x: x[1], reverse=True)
        }
            
    return result

def delete_topic(subject: str, topic: str):
    """Remove a topic's score data from the global knowledge profile."""
    data = load_global_profile()
    subj_key = subject.title().strip()
    topic_key = topic.title().strip()
    changed = False

    if subj_key in data and topic_key in data[subj_key]:
        del data[subj_key][topic_key]
        # Clean up empty subject entry
        if not data[subj_key]:
            del data[subj_key]
        changed = True

    if changed:
        save_global_profile(data)