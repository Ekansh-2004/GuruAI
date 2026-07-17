import sqlite3
import os
from contextlib import contextmanager

DB_FILE = "scholar.db"

@contextmanager
def get_db():
    """Returns a SQLite connection that auto-closes on exit. Always use with `with get_db() as conn:`."""
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initializes the database schema if tables do not exist."""
    schema = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        name TEXT NOT NULL,
        size INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS knowledge_profile (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        subject TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, subject)
    );

    CREATE TABLE IF NOT EXISTS user_memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        memory_item TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, memory_item)
    );

    CREATE TABLE IF NOT EXISTS memory_chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """
    with get_db() as conn:
        conn.executescript(schema)
        
        # ── Performance Indexes ──
        # Index every foreign-key column used in WHERE clauses.
        # Without these, every query does a full table scan.
        indexes = """
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id        ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_messages_session_id      ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_documents_session_id     ON documents(session_id);
        CREATE INDEX IF NOT EXISTS idx_memory_chat_user_id      ON memory_chat_history(user_id);
        CREATE INDEX IF NOT EXISTS idx_knowledge_profile_user   ON knowledge_profile(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_memories_user_id    ON user_memories(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_subjects_user_id    ON user_subjects(user_id);
        
        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_session_name ON documents(session_id, name);
        """
        conn.executescript(indexes)
        conn.commit()
