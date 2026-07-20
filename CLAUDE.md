# GuruAI: Development Guide

## Project Overview
GuruAI (internally "The Scholar") is a FastAPI + RAG tutoring system with:
- Corrective RAG (CRAG) chat over user-uploaded documents
- Hybrid retrieval (dense FAISS + sparse TF-IDF, fused via RRF)
- Adaptive mastery tracking (EMA algorithm) and a spaced-repetition scheduler
- LLM-generated quizzes, both session-wide and per-topic
- Persistent per-user memory and subject profiles
- Multi-user support with cookie-based auth

## Stack
- **Backend**: FastAPI, Python 3.x, uvicorn
- **Orchestration**: LangChain
- **LLMs**: Groq (`llama-3.3-70b-versatile`) for answers/quizzes; Google Gemini for the CRAG relevance grader
- **Embeddings**: HuggingFace `sentence-transformers`
- **Vector DB**: FAISS (CPU), persisted to disk per session
- **Storage**: SQLite (`scholar.db`)
- **Auth**: PBKDF2-HMAC-SHA256 password hashing + hand-rolled HMAC-signed tokens (stdlib only — no `bcrypt`, no `PyJWT`)
- **Frontend**: HTML5, vanilla JS, TailwindCSS (CDN), no build step
- **Docs**: PyPDF, docx2txt

## Directory Structure
```
server.py                  # thin entry point: app, middleware, include_router (~60 lines)
requirements.txt
requirements-dev.txt
src/
├── api/                   # HTTP layer
│   ├── deps.py            # get_current_user, check_auth_html, verify_session_ownership, set_auth_cookie
│   ├── schemas.py         # all Pydantic request models
│   ├── retriever_cache.py # LRU cache of per-session hybrid retrievers (get/refresh/invalidate)
│   └── routers/           # one module per functional area
│       ├── auth.py        # /api/auth/*
│       ├── pages.py       # HTML page serving + redirects
│       ├── sessions.py    # /api/sessions/* incl. upload, documents, knowledge
│       ├── chat.py        # /api/chat (SSE)
│       ├── quiz.py        # session quiz + /api/quiz/answer
│       ├── profile.py     # /api/profile/*
│       ├── srs.py         # review-queue, mark-reviewed, topics/statistics
│       ├── subjects.py    # /api/subjects/*
│       ├── topic.py       # /api/topic/explain, /api/topic/quiz
│       ├── memory.py      # /api/memory/*
│       └── user.py        # /api/user/profile, /api/user/stats
├── core/
│   ├── config.py          # env vars + chunking constants
│   ├── database.py        # DB_FILE, get_db() contextmanager, init_db() schema
│   └── llm.py             # shared ChatGroq singletons (default/balanced/creative)
├── rag/
│   ├── crag.py            # CRAG: relevance grading + web-search fallback
│   ├── chain.py           # builds the streaming answer chain
│   ├── retriever.py       # HybridRetriever, RRF fusion, per-document diversification
│   ├── embedder.py        # FAISS create/load/delete, get_db_path(), vectorstore_exists()
│   ├── loader.py          # PDF/DOCX/TXT → chunked Documents
│   ├── quiz.py            # session-wide quiz generation
│   └── topic_tutor.py     # per-topic explanations + quizzes
├── sessions/              # chat session persistence
│   ├── store.py           # sessions, messages, quiz JSON
│   └── documents.py       # uploaded-document metadata + knowledge-base clearing
├── personalization/
│   ├── mastery.py         # EMA, knowledge profile, SRS schedule views, review queue
│   ├── spaced_rep.py      # SpacedRepetitionScheduler
│   └── user_memory.py     # memory items, memory chat, subjects, user profile
├── auth/
│   └── auth.py            # hash_password, verify_password, create/verify_access_token
└── migrations/
    ├── 001_add_spaced_rep_columns.py
    ├── 002_add_document_metadata_columns.py
    └── 003_add_message_sources_column.py
static/
├── index.html             # main chat UI (~2340 lines, JS/CSS inline)
├── knowledge.html         # knowledge profile dashboard
├── topic.html             # per-topic tutor view
├── user.html              # user profile / memory / subjects
├── login.html
├── theme.js               # shared theme engine — currently NOT referenced by any page
└── widgets/
    ├── review-queue.js
    ├── srs-stats.js
    └── study-tracker.js
tests/
├── smoke_test.py          # end-to-end characterization test (no LLM needed)
└── routes_snapshot.json   # expected route table; guards against routes lost in refactors
scratch/                   # integration tests + benchmark (see Testing)
faiss_index_db/            # per-session FAISS indexes (gitignored)
```

Note: `src/auth/` and `src/personalization/` have no `__init__.py`. They resolve as
PEP 420 namespace packages. `src/`, `src/core/`, `src/rag/`, and `src/migrations/` do have one.

## Key Files & Functions
- `server.py` — creates the app and includes routers; no handler logic
- `api/deps.py` — `get_current_user` (401), `check_auth_html` (redirect), `verify_session_ownership` (403/404), `set_auth_cookie`
- `api/retriever_cache.py` — `get()` / `refresh()` / `invalidate()`; LRU, max 32 entries
- `sessions/store.py` — session + message CRUD, `save_quiz()` / `get_quiz()`
- `sessions/documents.py` — `add_document()`, `get_session_documents()`, `clear_session_knowledge_base()`
- `personalization/mastery.py` — `update_ema()`, `update_topic_performance()`, `get_performance_areas()`, `list_topics_with_schedule()`, `build_review_queue()`, `get_topic_statistics()`, `get_user_stats()`, `build_profile_summary()`
- `spaced_rep.py` — `SpacedRepetitionScheduler` class
- `quiz.py` — `generate_quiz_for_session_db()`
- `topic_tutor.py` — `generate_topic_explanation()`, `generate_topic_quiz()`
- `embedder.py` — `create_vectorstore()`, `load_existing_vectorstore()`, `get_db_path()` (module-level functions, not a class)
- `crag.py` — `build_crag_context()` returns `(context_text, source_label, sources_metadata)`
- `retriever.py` — `HybridRetriever` class, `reciprocal_rank_fusion()`

## API Routes
All JSON endpoints are under `/api/` and require a valid `access_token` cookie
via `Depends(get_current_user)`. HTML page routes redirect to `/login.html` instead
of returning 401.

**Auth** — `POST /api/auth/register|login|logout`
**Pages** — `GET /`, `/index.html`, `/knowledge.html`, `/topic.html`, `/user.html`, `/login.html`
**Sessions** — `GET|POST /api/sessions`, `DELETE /api/sessions/{id}`, `PATCH /api/sessions/{id}/title`, `GET /api/sessions/{id}/messages`, `GET /api/sessions/{id}/db-status`
**Documents** — `POST /api/sessions/{id}/upload` (multi-file), `GET /api/sessions/{id}/documents`, `DELETE /api/sessions/{id}/knowledge`
**Chat** — `POST /api/chat` (SSE stream; emits `{chunk}` frames, then `{sources}`, then `[DONE]`)
**Quiz** — `POST|GET /api/sessions/{id}/quiz`, `POST /api/quiz/answer`
**Profile** — `GET /api/profile`, `DELETE /api/profile/{subject}/{topic}`
**SRS** — `GET /api/suggestions/review-queue`, `POST /api/topics/{id}/mark-reviewed`, `GET /api/topics/statistics`
**Subjects** — `GET|POST /api/subjects`, `DELETE /api/subjects/{subject}`
**Topic tutor** — `POST /api/topic/explain`, `POST /api/topic/quiz`
**Memory** — `GET|POST|DELETE /api/memory`, `DELETE /api/memory/{index}`, `GET|POST /api/memory/chat`
**User** — `GET|POST /api/user/profile`, `GET /api/user/stats`

## Database Schema (SQLite — `scholar.db`)
Defined in `src/core/database.py::init_db()`.

- `users` — id, username (unique), password_hash, name, bio, created_at
- `sessions` — id (TEXT uuid), user_id, title, quiz_json, created_at, updated_at
- `messages` — id, session_id, role, content, sources, created_at
- `documents` — id, session_id, doc_id, name, size, file_type, status, storage_path, chunk_count, error, created_at
- `knowledge_profile` — id, user_id, subject, topic, correct, total, ema_score, last_reviewed_at, review_interval_days, review_count, next_review_date, updated_at; UNIQUE(user_id, subject, topic)
- `user_subjects` — id, user_id, subject, created_at; UNIQUE(user_id, subject)
- `user_memories` — id, user_id, memory_item, created_at; UNIQUE(user_id, memory_item)
- `memory_chat_history` — id, user_id, role, content, created_at

All child tables cascade on user/session delete. Indexes are created on every FK
column used in a WHERE clause.

## Environment
Required in `.env`:
- `GOOGLE_API_KEY` — Gemini, CRAG grader
- `GROQ_API_KEY` — Llama, answer + quiz chains
- `JWT_SECRET_KEY` — token signing; **falls back to a hardcoded default if unset**

## Coding Conventions
- Type hints on all functions
- FastAPI route docstrings describe inputs/outputs
- Functions and methods use snake_case; Python modules snake_case
- Frontend is vanilla JS, no framework, no build step; widget files use kebab-case
- CSS uses TailwindCSS utility classes

## How to Run
```bash
pip install -r requirements.txt
python server.py          # http://localhost:8000
```

## Testing
The fast, hermetic check — no API keys, runs against a temp DB and temp FAISS dir:
```bash
python tests/smoke_test.py     # 84 checks across every LLM-free endpoint
```
It also diffs the live route table against `tests/routes_snapshot.json`, so a route
accidentally dropped during a refactor fails loudly. Regenerate the snapshot
(by deleting it and re-running) only when routes intentionally change.

Heavier integration tests live in `scratch/`:
```bash
python scratch/test_spaced_rep.py        # no API keys needed
python scratch/test_multi_doc_upload.py  # downloads the embedding model
python scratch/test_multi_doc_retrieval.py  # needs GOOGLE_API_KEY + GROQ_API_KEY
```
These hit live LLM APIs and the real `scholar.db` — they are not hermetic and
require valid API keys. `scratch/benchmark.py` is a retrieval timing benchmark,
not a test, and needs `requirements-dev.txt` for numpy.

## Known Limitations / Tech Debt
- `static/index.html` is ~2340 lines with JS/CSS inline; all 5 pages need their JS/CSS extracted
- `static/theme.js` is dead; `toggleTheme` is duplicated inline across all 5 pages
- `escapeHtml` and `loadProfile` are duplicated across pages/widgets; ~45 raw `fetch()` calls each hand-roll error handling
- Migration modules start with digits, so `python -m src.migrations.001_...` fails; there is no applied-migration tracking
- CORS is `allow_origins=["*"]` alongside cookie auth
- `@app.on_event("startup")` is deprecated in favor of `lifespan`
- Logging is via `print()` throughout
- The `ddgs` web-search fallback in `crag.py` is inside a try/except and is not installed by default — it fails silently
- Vector indices are local only (no cloud backup)
