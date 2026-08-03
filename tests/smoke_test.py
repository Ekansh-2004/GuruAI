"""End-to-end characterization test for the GuruAI HTTP surface.

Runs against a throwaway Postgres database (created and dropped for this run
only), so it never touches the real app database. Exercises every endpoint
that does not require a live LLM call, plus a full route-table snapshot.

Requires a reachable Postgres server with the pgvector extension available
(set via DATABASE_URL, or defaults to the local dev instance used throughout
this migration: postgresql://guruai:guruai_dev_pw@localhost:5432/guruai_dev).
This is a real change from the old SQLite-backed version of this test, which
needed zero external services — worth knowing if this test is ever wired
into CI.

Run with:  python tests/smoke_test.py
Exits non-zero on the first failure.
"""
import json
import os
import sys
import uuid
from urllib.parse import urlparse, urlunparse

import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Create a throwaway Postgres database BEFORE importing the app ──
_BASE_DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://guruai:guruai_dev_pw@localhost:5432/guruai_dev"
)
_parsed = urlparse(_BASE_DATABASE_URL)
_TEST_DB_NAME = f"test_{uuid.uuid4().hex[:12]}"
_ADMIN_URL = urlunparse(_parsed._replace(path="/postgres"))
_TEST_DB_URL = urlunparse(_parsed._replace(path=f"/{_TEST_DB_NAME}"))


def _admin_execute(sql: str) -> None:
    conn = psycopg2.connect(_ADMIN_URL)
    conn.autocommit = True  # CREATE DATABASE / DROP DATABASE can't run inside a transaction
    try:
        conn.cursor().execute(sql)
    finally:
        conn.close()


_admin_execute(f'CREATE DATABASE "{_TEST_DB_NAME}"')

# get_db() registers the pgvector type on every connection, which fails if the
# extension doesn't exist yet — create it here, before any app code connects.
_bootstrap_conn = psycopg2.connect(_TEST_DB_URL)
_bootstrap_conn.autocommit = True
_bootstrap_conn.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector;")
_bootstrap_conn.close()

import src.core.database as database  # noqa: E402
database.DATABASE_URL = _TEST_DB_URL

from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402

database.init_db()

_failures = []
_checks = 0


def check(label, actual, expected):
    global _checks
    _checks += 1
    if actual != expected:
        _failures.append(f"{label}\n     expected: {expected!r}\n     actual:   {actual!r}")
        print(f"  FAIL  {label}")
    else:
        print(f"  ok    {label}")


def check_true(label, cond, detail=""):
    check(label + (f" ({detail})" if detail else ""), bool(cond), True)


def section(name):
    print(f"\n=== {name} ===")


client = TestClient(server.app)

# ── Auth ──────────────────────────────────────────────────────────────
section("auth")
r = client.post("/api/auth/register", json={"username": "Alice", "password": "pw123"})
check("register 200", r.status_code, 200)
check("register status ok", r.json()["status"], "ok")
check_true("auth cookie set", "access_token" in r.cookies or "access_token" in client.cookies)
USER_ID = r.json()["user_id"]

r = client.post("/api/auth/register", json={"username": "alice", "password": "pw123"})
check("duplicate username rejected (case-normalized)", r.status_code, 400)

r = client.post("/api/auth/register", json={"username": "", "password": "pw"})
check("empty username rejected", r.status_code, 400)

r = client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
check("bad password rejected", r.status_code, 400)

r = client.post("/api/auth/login", json={"username": "ALICE", "password": "pw123"})
check("login 200 (case-insensitive)", r.status_code, 200)

# ── Unauthenticated access ────────────────────────────────────────────
section("auth gating")
anon = TestClient(server.app)
check("anon /api/sessions -> 401", anon.get("/api/sessions").status_code, 401)
check("anon /api/profile -> 401", anon.get("/api/profile").status_code, 401)
check("anon /api/user/stats -> 401", anon.get("/api/user/stats").status_code, 401)
r = anon.get("/", follow_redirects=False)
check("anon / -> redirect", r.status_code, 307)
check("anon / -> /login.html", r.headers["location"], "/login.html")
r = client.get("/login.html", follow_redirects=False)
check("logged-in /login.html -> redirect", r.status_code, 307)
check("logged-in /login.html -> /index.html", r.headers["location"], "/index.html")
r = client.get("/", follow_redirects=False)
check("logged-in / -> 200", r.status_code, 200)

# ── Sessions ──────────────────────────────────────────────────────────
section("sessions")
check("sessions initially empty", client.get("/api/sessions").json(), {})

r = client.post("/api/sessions")
check("create session 200", r.status_code, 200)
SID = r.json()["session_id"]
check_true("session_id is uuid-shaped", len(SID) == 36)

sessions = client.get("/api/sessions").json()
check("one session listed", len(sessions), 1)
check("default title", sessions[SID]["title"], "New Chat")
check("no messages yet", sessions[SID]["messages"], [])
check("no documents yet", sessions[SID]["documents"], [])

r = client.patch(f"/api/sessions/{SID}/title", data={"title": "Renamed"})
check("rename 200", r.status_code, 200)
check("title persisted", client.get("/api/sessions").json()[SID]["title"], "Renamed")

check("messages empty", client.get(f"/api/sessions/{SID}/messages").json(), [])
check("db-status false", client.get(f"/api/sessions/{SID}/db-status").json(), {"exists": False})
check("documents empty", client.get(f"/api/sessions/{SID}/documents").json(), [])
check("quiz empty", client.get(f"/api/sessions/{SID}/quiz").json(), {})

# ── Cross-user isolation ──────────────────────────────────────────────
section("cross-user isolation")
other = TestClient(server.app)
other.post("/api/auth/register", json={"username": "bob", "password": "pw"})
check("bob sees no sessions", other.get("/api/sessions").json(), {})
check("bob cannot read alice's messages", other.get(f"/api/sessions/{SID}/messages").status_code, 403)
check("bob cannot rename alice's session",
      other.patch(f"/api/sessions/{SID}/title", data={"title": "hax"}).status_code, 403)
check("bob cannot delete alice's session", other.delete(f"/api/sessions/{SID}").status_code, 403)
check("missing session -> 404", client.get("/api/sessions/no-such-id/messages").status_code, 404)

# ── Quiz answers drive the knowledge profile ──────────────────────────
section("quiz answers -> profile")
check("profile initially empty", client.get("/api/profile").json(), {})

for correct in (True, True, False):
    r = client.post("/api/quiz/answer", json={
        "session_id": SID, "subject": "biology", "topic": "mitosis", "is_correct": correct,
    })
    check(f"submit answer correct={correct}", r.status_code, 200)

profile = client.get("/api/profile").json()
check("subject title-cased", list(profile.keys()), ["Biology"])
buckets = profile["Biology"]
tracked = [t for lvl in buckets.values() for t in lvl]
check("one topic tracked", len(tracked), 1)
check("topic title-cased", tracked[0][0], "Mitosis")
check("correct count", tracked[0][2], 2)
check("total count", tracked[0][3], 3)

stats = client.get("/api/user/stats").json()
check("total_questions", stats["total_questions"], 3)
check_true("average_mastery_pct is a number", isinstance(stats["average_mastery_pct"], (int, float)))

# ── Spaced repetition ─────────────────────────────────────────────────
section("spaced repetition")
tstats = client.get("/api/topics/statistics").json()
check("total_topics", tstats["total_topics"], 1)
check("strongest_topic", tstats["strongest_topic"], "Mitosis")

rq = client.get("/api/suggestions/review-queue").json()
check("review queue total_topics", rq["total_topics"], 1)
check_true("queue is a list", isinstance(rq["queue"], list))
check("bad category rejected",
      client.get("/api/suggestions/review-queue", params={"category": "bogus"}).status_code, 422)
check("bad sort rejected",
      client.get("/api/suggestions/review-queue", params={"sort": "bogus"}).status_code, 422)
check("limit out of range rejected",
      client.get("/api/suggestions/review-queue", params={"limit": 999}).status_code, 422)

# Find the topic id so we can mark it reviewed
topic_id = None
for lvl in client.get("/api/profile").json()["Biology"].values():
    if lvl:
        from src.personalization import mastery as _m
        for row in _m.list_topics_with_schedule(USER_ID):
            if row["topic"] == "Mitosis":
                topic_id = row["id"]
check_true("found topic id", topic_id is not None)

r = client.post(f"/api/topics/{topic_id}/mark-reviewed", json={"score": 8})
check("mark-reviewed 200", r.status_code, 200)
check("mark-reviewed topic", r.json()["topic"], "Mitosis")
check_true("next_review returned", r.json()["next_review"] is not None)
check("score > 10 rejected",
      client.post(f"/api/topics/{topic_id}/mark-reviewed", json={"score": 11}).status_code, 400)
check("score < 0 rejected",
      client.post(f"/api/topics/{topic_id}/mark-reviewed", json={"score": -1}).status_code, 400)
check("unknown topic -> 404",
      client.post("/api/topics/999999/mark-reviewed", json={"score": 5}).status_code, 404)
check("bob cannot review alice's topic",
      other.post(f"/api/topics/{topic_id}/mark-reviewed", json={"score": 5}).status_code, 404)

# ── Subjects ──────────────────────────────────────────────────────────
section("subjects")
check("no subjects initially", client.get("/api/subjects").json(), {"subjects": []})
r = client.post("/api/subjects", json={"subject": "Chemistry"})
check("add subject", r.json()["subjects"], ["Chemistry"])
client.post("/api/subjects", json={"subject": "Physics"})
check("two subjects", sorted(client.get("/api/subjects").json()["subjects"]), ["Chemistry", "Physics"])
r = client.delete("/api/subjects/Chemistry")
check("delete subject", r.json()["subjects"], ["Physics"])
check("bob's subjects isolated", other.get("/api/subjects").json(), {"subjects": []})

# ── Memory (DB-only paths; extraction needs an LLM) ───────────────────
section("memory")
check("memory initially empty", client.get("/api/memory").json(), {"items": []})
check("memory chat history empty", client.get("/api/memory/chat").json(), {"history": []})
import src.personalization.user_memory as um
um.add_memory_items(USER_ID, ["prefers worked examples", "studying for finals"])
check("two memory items", len(client.get("/api/memory").json()["items"]), 2)
r = client.delete("/api/memory/0")
check("delete by index leaves one", len(r.json()["items"]), 1)
check("clear all", client.delete("/api/memory").json(), {"items": []})

# ── User profile ──────────────────────────────────────────────────────
section("user profile")
prof = client.get("/api/user/profile").json()
check("default name", prof["name"], "The Scholar")
r = client.post("/api/user/profile", json={"name": "Alice A", "bio": "bio text"})
check("profile saved name", r.json()["name"], "Alice A")
check("profile persisted", client.get("/api/user/profile").json()["bio"], "bio text")

# ── Profile topic deletion ────────────────────────────────────────────
section("profile deletion")
r = client.delete("/api/profile/Biology/Mitosis")
check("delete topic 200", r.status_code, 200)
check("profile empty after delete", client.get("/api/profile").json(), {})

# ── Chat/quiz guards (no LLM call — these fail before reaching one) ───
section("llm-free guards")
check("chat without vectorstore -> 400",
      client.post("/api/chat", json={"session_id": SID, "question": "hi"}).status_code, 400)
check("quiz without vectorstore -> 400",
      client.post(f"/api/sessions/{SID}/quiz").status_code, 400)
check("bob cannot chat in alice's session",
      other.post("/api/chat", json={"session_id": SID, "question": "hi"}).status_code, 403)

# ── Session deletion cascades ─────────────────────────────────────────
section("session deletion")
from src.sessions import store as _store
_store.add_message(SID, "user", "hello")
check("message added", len(client.get(f"/api/sessions/{SID}/messages").json()), 1)
check("delete session", client.delete(f"/api/sessions/{SID}").json(), {"status": "deleted"})
check("sessions empty after delete", client.get("/api/sessions").json(), {})
check("messages 404 after delete", client.get(f"/api/sessions/{SID}/messages").status_code, 404)

# ── Logout ────────────────────────────────────────────────────────────
section("logout")
check("logout 200", client.post("/api/auth/logout").status_code, 200)
check("sessions 401 after logout", client.get("/api/sessions").status_code, 401)

# ── Route table snapshot ──────────────────────────────────────────────
section("route table")
routes = sorted(
    (r.path, ",".join(sorted(getattr(r, "methods", []) or [])))
    for r in server.app.routes
)
snapshot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routes_snapshot.json")
if os.path.exists(snapshot):
    expected = [tuple(x) for x in json.load(open(snapshot))]
    check("route table unchanged", routes, expected)
    if routes != expected:
        for missing in sorted(set(expected) - set(routes)):
            print(f"     MISSING: {missing}")
        for added in sorted(set(routes) - set(expected)):
            print(f"     ADDED:   {added}")
else:
    json.dump(routes, open(snapshot, "w"), indent=1)
    print(f"  ..    wrote baseline snapshot ({len(routes)} routes)")

# ── Report ────────────────────────────────────────────────────────────
_admin_execute(f'DROP DATABASE IF EXISTS "{_TEST_DB_NAME}"')
print("\n" + "=" * 60)
if _failures:
    print(f"FAILED — {len(_failures)} of {_checks} checks\n")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"PASSED — all {_checks} checks")
