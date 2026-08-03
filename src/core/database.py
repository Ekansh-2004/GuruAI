import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

# Load .env here too (not just in config.py) — this module reads DATABASE_URL
# at import time, and which module gets imported first (and therefore whose
# load_dotenv() call runs first) isn't guaranteed, so this can't rely on
# config.py having already loaded it.
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://guruai:guruai_dev_pw@localhost:5432/guruai_dev"
)

# 768 = the configured output_dimensionality of gemini-embedding-001 (src/rag/embedder.py).
EMBEDDING_DIM = 768

@contextmanager
def get_db():
    """Returns a Postgres connection that auto-closes on exit. Always use with `with get_db() as conn:`."""
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initializes the database schema if tables do not exist."""
    schema = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        name TEXT DEFAULT 'The Scholar',
        bio TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        quiz_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        sources TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        doc_id TEXT,
        name TEXT NOT NULL,
        size INTEGER NOT NULL,
        file_type TEXT,
        status TEXT NOT NULL DEFAULT 'ready',
        storage_path TEXT,
        chunk_count INTEGER DEFAULT 0,
        error TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS knowledge_profile (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        subject TEXT NOT NULL,
        topic TEXT NOT NULL,
        correct INTEGER DEFAULT 0,
        total INTEGER DEFAULT 0,
        ema_score REAL,
        last_reviewed_at TIMESTAMP DEFAULT NULL,
        review_interval_days INTEGER DEFAULT 1,
        review_count INTEGER DEFAULT 0,
        next_review_date TIMESTAMP DEFAULT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, subject, topic)
    );

    CREATE TABLE IF NOT EXISTS user_subjects (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        subject TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, subject)
    );

    CREATE TABLE IF NOT EXISTS user_memories (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        memory_item TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, memory_item)
    );

    CREATE TABLE IF NOT EXISTS memory_chat_history (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    -- Replaces the local per-session FAISS index: one row per chunk, embedding
    -- stored directly alongside its text and source metadata (see src/rag/embedder.py).
    CREATE TABLE IF NOT EXISTS document_chunks (
        id SERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        document_id TEXT,
        content TEXT NOT NULL,
        metadata JSONB,
        embedding vector({EMBEDDING_DIM}),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );

    -- Performance indexes: every foreign-key column used in a WHERE clause.
    CREATE INDEX IF NOT EXISTS idx_sessions_user_id        ON sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_messages_session_id      ON messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_documents_session_id     ON documents(session_id);
    CREATE INDEX IF NOT EXISTS idx_memory_chat_user_id      ON memory_chat_history(user_id);
    CREATE INDEX IF NOT EXISTS idx_knowledge_profile_user   ON knowledge_profile(user_id);
    CREATE INDEX IF NOT EXISTS idx_user_memories_user_id    ON user_memories(user_id);
    CREATE INDEX IF NOT EXISTS idx_user_subjects_user_id    ON user_subjects(user_id);
    CREATE INDEX IF NOT EXISTS idx_document_chunks_session_id ON document_chunks(session_id);

    CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_session_name ON documents(session_id, name);
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(schema)
        conn.commit()
