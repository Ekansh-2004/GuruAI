
import streamlit as st
from typing import List, Dict
import json
import os

HISTORY_FILE = "chat_history.json"

def load_history() -> List[Dict[str, str]]:
    """Load persistent history from JSON."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history: List[Dict[str, str]]):
    """Save history to JSON."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def add_message(role: str, content: str):
    """Add user/assistant message."""
    history = load_history()
    history.append({"role": role, "content": content})
    save_history(history)
    return history

def clear_history():
    """Clear all history."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return []

def get_recent_weak_topics(n=3) -> List[str]:
    """Simple analysis: extract repeated topics (future personalization)."""
    history = load_history()
    user_questions = [msg["content"].lower() for msg in history if msg["role"] == "user"]
    words = " ".join(user_questions).split()
    from collections import Counter
    common = Counter(words).most_common(n)
    return [word for word, count in common if count > 1]  # Repeated words = weak?