"""GuruAI FastAPI application entry point.

Route handlers live in src/api/routers/, one module per functional area.
Shared auth dependencies are in src/api/deps.py.

Run with:  python server.py
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import (
    auth,
    chat,
    memory,
    pages,
    profile,
    quiz,
    sessions,
    srs,
    subjects,
    topic,
    user,
)
from src.core.database import init_db

app = FastAPI(title="GuruAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Initialize the SQLite schema before serving traffic."""
    init_db()


app.mount("/static", StaticFiles(directory="static"), name="static")

# JSON API. Every router below gates on Depends(get_current_user).
app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(quiz.router)
app.include_router(profile.router)
app.include_router(srs.router)
app.include_router(subjects.router)
app.include_router(topic.router)
app.include_router(memory.router)
app.include_router(user.router)

# HTML pages last, so the /api/* routes above always take precedence.
app.include_router(pages.router)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
