# GuruAI: Spaced Repetition Development Guide

## Project Overview
GuruAI is a FastAPI + RAG tutoring system with:
- Adaptive mastery tracking (EMA algorithm)
- Vector search (FAISS)
- Quiz generation
- Multi-user support with JWT auth

## Current Stack
- **Backend**: FastAPI, Python 3.x
- **AI**: LangChain, Google GenAI, Groq LLM
- **Vector DB**: FAISS (in-memory, CPU)
- **Frontend**: HTML5, Vanilla JS, TailwindCSS
- **Auth**: SQLite, bcrypt, JWT
- **Docs**: PyPDF, docx2txt

## Directory Structure
src/
├── rag/
│   ├── crag.py          # Corrective RAG pipeline
│   ├── faiss_embedder.py # Vector embeddings
│   ├── quiz_gen.py      # Dynamic quiz generation
│   └── doc_loader.py    # PDF/DOCX processing
├── personalization/
│   ├── mastery_tracker.py    # EMA algorithm
│   └── user_memory.py        # Persistent memory system
├── auth/
│   ├── auth.py          # JWT, bcrypt
│   └── user.py          # User model
static/
├── index.html
├── style.css
├── main.js
├── widgets/
└── dark-mode.js
server.py               # FastAPI entry point
requirements.txt
.env.example

## Key Files & Functions
- `server.py`: Main FastAPI app, routes: `/chat`, `/quiz`, `/upload`
- `mastery_tracker.py`: `MasteryTracker` class with `update_ema()` method
- `quiz_gen.py`: `generate_quiz()` function
- `faiss_embedder.py`: `FAISSEmbedder` class

## Database Schema (SQLite)
users: id, username, password_hash, created_at
sessions: id, user_id, doc_path, vector_index_path, created_at
mastery: id, user_id, topic, mastery_level (0-1), last_updated
quiz_history: id, user_id, topic, score, timestamp

## Coding Conventions
- Type hints on all functions
- FastAPI route docstrings describe inputs/outputs
- Class methods use snake_case
- Frontend uses vanilla JS, no frameworks (yet)
- CSS uses TailwindCSS utility classes

## Current Limitations (for future ref)
- Vector indices stored locally (no cloud backup)
- No real-time collaboration yet
- Mobile responsive but not optimized

## How to Run
```bash
python server.py
# Opens at http://localhost:8000
```