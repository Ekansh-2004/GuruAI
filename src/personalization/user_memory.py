"""
User Memory — persistent cross-session preferences and facts about the student.

Two responsibilities:
1. Storage + LLM extraction of user preferences (injected into all study sessions)
2. Conversational memory chat — a dedicated LLM chat that learns about the user
   and responds warmly while silently extracting and persisting preferences.
"""

import json
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List
from src.core.config import GROQ_API_KEY

MEMORY_FILE = "user_memory.json"


# ── Storage ───────────────────────────────────────────────────────────────────

def _load_data() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {"items": [], "chat_history": []}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def _save_data(data: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_memory() -> list[str]:
    return _load_data().get("items", [])


def save_memory(items: list[str]):
    data = _load_data()
    data["items"] = items
    _save_data(data)


def delete_memory_item(index: int) -> list[str]:
    items = load_memory()
    if 0 <= index < len(items):
        items.pop(index)
        save_memory(items)
    return load_memory()


def add_memory_items(new_items: list[str]) -> list[str]:
    items = load_memory()
    for item in new_items:
        if item and item.strip() and item not in items:
            items.append(item)
    save_memory(items)
    return items


def clear_all_memory():
    data = _load_data()
    data["items"] = []
    _save_data(data)


# ── Memory Chat History ───────────────────────────────────────────────────────

def get_chat_history() -> list[dict]:
    return _load_data().get("chat_history", [])


def append_chat_message(role: str, content: str):
    data = _load_data()
    if "chat_history" not in data:
        data["chat_history"] = []
    data["chat_history"].append({"role": role, "content": content})
    _save_data(data)


# ── LLM Extraction ────────────────────────────────────────────────────────────

class ExtractedPreferences(BaseModel):
    preferences: List[str] = Field(
        description="List of concise preference/fact statements about the user, max 12 words each."
    )


EXTRACT_PROMPT = PromptTemplate(
    template="""You are a user-preference extraction system for a CS tutoring AI.

Extract any persistent preferences, learning styles, technical choices, or personal facts
from the student's message that would help a tutor serve them better.

Rules:
- Each item: concise statement starting with "User" (max 12 words)
  e.g. "User prefers Python for all code examples"
  e.g. "User is a 3rd year Computer Science student"
- Only extract meaningful, actionable, persistent preferences
- Return empty list if nothing extractable
- Do NOT duplicate already stored preferences

Already stored:
{existing}

Student's message: "{message}"

{format_instructions}""",
    input_variables=["message", "existing"],
    partial_variables={},
)


def extract_preferences_from_message(message: str) -> list[str]:
    model = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, max_retries=0)
    parser = JsonOutputParser(pydantic_object=ExtractedPreferences)
    prompt = EXTRACT_PROMPT.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser
    existing = load_memory()
    existing_str = "\n".join(f"- {i}" for i in existing) if existing else "None"
    try:
        result = chain.invoke({"message": message, "existing": existing_str})
        return result.get("preferences", [])
    except Exception as e:
        print(f"Memory extraction error: {e}")
        return []


# ── Conversational Memory Chat ────────────────────────────────────────────────

MEMORY_CHAT_SYSTEM = """You are the student's personal Study Companion — a warm, friendly AI whose only job in this chat is to learn about the student and remember their preferences.

Your role here:
- Have a natural, friendly conversation to understand the student's preferences, background, and learning style
- When the student shares preferences (e.g. "I like Python", "explain with analogies"), warmly acknowledge them and confirm you'll remember
- Keep responses SHORT (2-4 sentences max) — this is a quick preference-setting chat, not a tutoring session
- If the student asks an academic question, politely redirect: "I'll note that! For actual study help, head to a Study Session — I'll apply your preferences there."
- Do not give long explanations or teach here

Current stored preferences (for your awareness, don't re-ask for these):
{stored_preferences}

Be warm, brief, and confirmatory."""


def memory_chat(user_message: str) -> tuple[str, list[str]]:
    """
    Send a message to the memory chat LLM. Returns (bot_reply, newly_extracted_items).
    Also persists the conversation and any extracted preferences.
    """
    # 1. Get history
    history = get_chat_history()
    append_chat_message("user", user_message)

    # 2. Build formatted history for LLM
    formatted_history = []
    for h in history:
        if h["role"] == "user":
            formatted_history.append(HumanMessage(content=h["content"]))
        else:
            formatted_history.append(AIMessage(content=h["content"]))

    stored = load_memory()
    stored_str = "\n".join(f"- {i}" for i in stored) if stored else "None yet"

    # 3. Call LLM for conversational response
    model = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, max_retries=0, streaming=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_CHAT_SYSTEM),
        *[(("human" if m.type == "human" else "ai"), m.content) for m in formatted_history],
        ("human", "{message}"),
    ])
    chain = prompt | model | StrOutputParser()
    reply = chain.invoke({"stored_preferences": stored_str, "message": user_message})

    # 4. Persist bot reply
    append_chat_message("assistant", reply)

    # 5. Extract preferences silently
    extracted = extract_preferences_from_message(user_message)
    if extracted:
        add_memory_items(extracted)

    return reply, extracted


# ── System Context for Study Sessions ─────────────────────────────────────────

def get_memory_as_system_context() -> str:
    items = load_memory()
    if not items:
        return ""
    lines = "\n".join(f"- {item}" for item in items)
    return f"""PERSISTENT USER PREFERENCES (apply these in every response without being asked):
{lines}"""


# ── Subject Management ─────────────────────────────────────────────────────────

def load_subjects() -> list[str]:
    """Return the list of subjects the student has registered."""
    return _load_data().get("subjects", [])


def save_subject(subject: str) -> list[str]:
    """Add a subject if not already present. Returns updated list."""
    data = _load_data()
    subjects = data.get("subjects", [])
    subject = subject.strip().title()
    if subject and subject not in subjects:
        subjects.append(subject)
        data["subjects"] = subjects
        _save_data(data)
    return subjects


def delete_subject(subject: str) -> list[str]:
    """Remove a subject by name. Returns updated list."""
    data = _load_data()
    subjects = data.get("subjects", [])
    subject = subject.strip().title()
    if subject in subjects:
        subjects.remove(subject)
        data["subjects"] = subjects
        _save_data(data)
    return subjects


def get_subjects_prompt_constraint() -> str:
    """Return a formatted string to inject into LLM prompts restricting subject classification."""
    subjects = load_subjects()
    if not subjects:
        return ""
    formatted = ", ".join(f'"{s}"' for s in subjects)
    return f"""SUBJECT CLASSIFICATION CONSTRAINT:
The student is currently studying these subjects ONLY: {formatted}.
You MUST map every topic/question to one of these exact subject names.
Do NOT use any other subject name outside this list."""
