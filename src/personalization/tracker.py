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
        "documents": [], # <-- ADD THIS LINE
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

def update_topic_performance(session_id: str, subject: str, topic: str, correct: bool):
    """Record quiz performance in the session database so it can be dynamically subtracted when deleted."""
    data = load_all_sessions()
    if session_id not in data:
        return
        
    session_data = data[session_id]
    if "topic_scores" not in session_data:
        session_data["topic_scores"] = {}
        
    subj = subject.title().strip()
    if subj not in session_data["topic_scores"]:
        session_data["topic_scores"][subj] = {}
        
    t = topic.title().strip()
    if t not in session_data["topic_scores"][subj]:
        session_data["topic_scores"][subj][t] = {"correct": 0, "total": 0}
        
    session_data["topic_scores"][subj][t]["total"] += 1
    if correct:
        session_data["topic_scores"][subj][t]["correct"] += 1
        
    save_all_sessions(data)

def get_performance_areas():
    """Aggregates all topic scores from all active sessions, classifying them dynamically into weak, average, and strong by subject."""
    data = load_all_sessions()
    global_subjects = {}
    
    # Aggregate from all valid sessions
    for sid, session in data.items():
        session_scores = session.get("topic_scores", {})
        for subj, topics in session_scores.items():
            if subj not in global_subjects:
                global_subjects[subj] = {}
            for t, stats in topics.items():
                if t not in global_subjects[subj]:
                    global_subjects[subj][t] = {"correct": 0, "total": 0}
                global_subjects[subj][t]["correct"] += stats["correct"]
                global_subjects[subj][t]["total"] += stats["total"]
    
    result = {}
    for subj, topics in global_subjects.items():
        weak, average, strong = [], [], []
        for t, stats in topics.items():
            if stats["total"] == 0: continue
            score = stats["correct"] / stats["total"]
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