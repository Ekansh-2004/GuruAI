"""Topic mastery tracking (EMA) and spaced-repetition scheduling.

Split out of the former src/personalization/tracker.py. Session/message storage
now lives in src/sessions/store.py and document metadata in
src/sessions/documents.py.

Two paths write mastery:
  - update_topic_performance(): driven by individual right/wrong quiz answers.
  - update_ema():               driven by a manual 0-1 self-rated review score.
Both advance the spaced-repetition schedule via the shared `scheduler`.
"""
import difflib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from src.core.database import get_db
from src.personalization.spaced_rep import SpacedRepetitionScheduler

# EMA smoothing factor
_EMA_ALPHA = 0.3

# Shared scheduler instance for computing spaced-repetition review dates.
scheduler = SpacedRepetitionScheduler()


@dataclass
class MasteryRecord:
    """A user's mastery record for one (subject, topic) pair, as stored in knowledge_profile."""
    user_id: int
    subject: str
    topic: str
    correct: int = 0
    total: int = 0
    ema_score: Optional[float] = None
    last_reviewed_at: Optional[datetime] = None
    review_interval_days: int = 1
    review_count: int = 0
    next_review_date: Optional[datetime] = None


# ── Global Student Profile / Knowledge Base ──

def load_global_profile(user_id: int) -> Dict:
    """Load the globally aggregated knowledge profile for a user from SQLite."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT subject, topic, correct, total, ema_score FROM knowledge_profile WHERE user_id = ?",
            (user_id,)
        )
        profile = {}
        for r in cur.fetchall():
            subj = r["subject"]
            topic = r["topic"]
            if subj not in profile:
                profile[subj] = {}
            profile[subj][topic] = {
                "correct": r["correct"],
                "total": r["total"],
                "ema_score": r["ema_score"]
            }
        return profile


def update_topic_performance(session_id: str, subject: str, topic: str, correct: bool):
    """Record quiz performance and update the EMA score for the topic globally in SQLite."""
    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")  # Write-lock to prevent concurrent stale reads
        cur = conn.cursor()
        # 1. Fetch user_id from the session
        cur.execute("SELECT user_id FROM sessions WHERE id = ?", (session_id,))
        sess_row = cur.fetchone()
        if not sess_row:
            return  # Session not found
        user_id = sess_row["user_id"]

        subj = subject.strip().title()
        t = topic.strip().title()

        # 2. Query existing topics under this subject for fuzzy matching
        cur.execute(
            "SELECT topic FROM knowledge_profile WHERE user_id = ? AND subject = ?",
            (user_id, subj)
        )
        existing_topics = [r["topic"] for r in cur.fetchall()]

        matches = difflib.get_close_matches(t, existing_topics, n=1, cutoff=0.85)
        if matches:
            t = matches[0]

        # 3. Retrieve stats
        cur.execute(
            "SELECT correct, total, ema_score FROM knowledge_profile WHERE user_id = ? AND subject = ? AND topic = ?",
            (user_id, subj, t)
        )
        row = cur.fetchone()
        if not row:
            correct_val = 0
            total_val = 0
            ema = None
        else:
            correct_val = row["correct"]
            total_val = row["total"]
            ema = row["ema_score"]

        total_val += 1
        if correct:
            correct_val += 1

        recent = 1.0 if correct else 0.0
        prev = ema if ema is not None else 0.0
        new_ema = (recent * _EMA_ALPHA) + (prev * (1 - _EMA_ALPHA))

        # 4. Advance the spaced-repetition schedule for this review
        cur.execute(
            "SELECT review_count FROM knowledge_profile WHERE user_id = ? AND subject = ? AND topic = ?",
            (user_id, subj, t)
        )
        prev_row = cur.fetchone()
        new_review_count = (prev_row["review_count"] or 0) + 1 if prev_row else 1
        now = datetime.now()
        next_review_date = scheduler.calculate_next_review_date(new_ema, new_review_count, now)
        review_interval_days = (next_review_date - now).days

        # 5. Insert or update entry
        conn.execute(
            """
            INSERT INTO knowledge_profile (
                user_id, subject, topic, correct, total, ema_score,
                last_reviewed_at, review_interval_days, review_count, next_review_date, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, subject, topic) DO UPDATE SET
                correct = excluded.correct,
                total = excluded.total,
                ema_score = excluded.ema_score,
                last_reviewed_at = excluded.last_reviewed_at,
                review_interval_days = excluded.review_interval_days,
                review_count = excluded.review_count,
                next_review_date = excluded.next_review_date,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                user_id, subj, t, correct_val, total_val, new_ema,
                now, review_interval_days, new_review_count, next_review_date
            )
        )
        conn.commit()


def get_performance_areas(user_id: int) -> Dict:
    """Returns globally tracked topics for a user classified into weak, average, and strong."""
    profile = load_global_profile(user_id)

    result = {}
    for subj, topics in profile.items():
        weak, average, strong = [], [], []
        for t, stats in topics.items():
            if stats["total"] == 0:
                continue
            score = stats["ema_score"] if stats.get("ema_score") is not None else (stats["correct"] / stats["total"])
            val = (t, score, stats["correct"], stats["total"])
            if score < 0.5:
                weak.append(val)
            elif score <= 0.75:
                average.append(val)
            else:
                strong.append(val)

        result[subj] = {
            "weak": sorted(weak, key=lambda x: x[1]),
            "average": sorted(average, key=lambda x: x[1]),
            "strong": sorted(strong, key=lambda x: x[1], reverse=True)
        }

    return result


def delete_topic(user_id: int, subject: str, topic: str):
    """Remove a topic's score data from the global knowledge profile in SQLite."""
    subj_key = subject.strip().title()
    topic_key = topic.strip().title()
    with get_db() as conn:
        conn.execute(
            "DELETE FROM knowledge_profile WHERE user_id = ? AND subject = ? AND topic = ?",
            (user_id, subj_key, topic_key)
        )
        conn.commit()


def build_profile_summary(user_id: int) -> str:
    """Serialize the global knowledge profile into a compact text block
    the LLM can read and reason about.
    """
    profile = get_performance_areas(user_id)
    if not profile:
        return ""
    lines = []
    for subject, levels in profile.items():
        subject_lines = []
        for level_name, items in levels.items():
            for item in items:
                topic, score = item[0], item[1]
                pct = round(score * 100)
                subject_lines.append(f"    - {level_name.capitalize()} ({pct}%): {topic}")
        if subject_lines:
            lines.append(f"  Subject: {subject}")
            lines.extend(subject_lines)
    return "\n".join(lines) if lines else ""


# ── Spaced-repetition schedule views ──

def _parse_ts(value) -> Optional[datetime]:
    """Parse a SQLite-stored timestamp string back into a datetime, or None."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


_TOPIC_COLUMNS = """
    id, subject, topic, correct, total, ema_score,
    last_reviewed_at, review_interval_days, review_count, next_review_date
"""


def _topic_row_to_dict(row) -> dict:
    """Shape a knowledge_profile row into the schedule dict shared by the SRS endpoints."""
    ema_score = row["ema_score"]
    mastery_level = ema_score if ema_score is not None else (
        row["correct"] / row["total"] if row["total"] else 0.0
    )
    last_reviewed_at = _parse_ts(row["last_reviewed_at"])
    next_review_date = _parse_ts(row["next_review_date"])
    days_until_review = (
        (next_review_date.date() - datetime.now().date()).days if next_review_date else 0
    )
    interval_days = row["review_interval_days"] or 1
    urgency_score = round(min(1.0, max(0.0, 1 - (days_until_review / interval_days))), 2)

    return {
        "id": row["id"],
        "subject": row["subject"],
        "topic": row["topic"],
        "mastery_level": mastery_level,
        "mastery_category": scheduler.get_mastery_category(mastery_level),
        "last_reviewed": last_reviewed_at.date().isoformat() if last_reviewed_at else None,
        "next_review": next_review_date.date().isoformat() if next_review_date else None,
        "days_until_review": days_until_review,
        "review_count": row["review_count"] or 0,
        "is_due": scheduler.is_due_for_review(last_reviewed_at, next_review_date),
        "urgency_score": urgency_score,
    }


def list_topics_with_schedule(user_id: int) -> List[dict]:
    """Return every tracked topic for a user with its spaced-repetition schedule info."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {_TOPIC_COLUMNS} FROM knowledge_profile WHERE user_id = ?",
            (user_id,)
        )
        rows = cur.fetchall()
    return [_topic_row_to_dict(row) for row in rows]


def get_topic_by_id(topic_id: int, user_id: int) -> Optional[dict]:
    """Fetch a single topic's schedule info, scoped to its owning user."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {_TOPIC_COLUMNS} FROM knowledge_profile WHERE id = ? AND user_id = ?",
            (topic_id, user_id)
        )
        row = cur.fetchone()
    return _topic_row_to_dict(row) if row else None


def update_ema(topic_id: int, user_id: int, score: float) -> Optional[dict]:
    """Record a manual review score (0.0-1.0) for a topic and advance its review schedule.

    Unlike update_topic_performance (driven by individual quiz answers), this sets the
    EMA directly from a normalized score, e.g. update_ema(topic_id, user_id, score_out_of_10 / 10).
    Returns the updated topic dict, or None if no such topic exists for this user.
    """
    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.cursor()
        cur.execute(
            "SELECT correct, total, ema_score, review_count FROM knowledge_profile WHERE id = ? AND user_id = ?",
            (topic_id, user_id)
        )
        row = cur.fetchone()
        if not row:
            return None

        prev_ema = row["ema_score"] if row["ema_score"] is not None else 0.0
        new_ema = (score * _EMA_ALPHA) + (prev_ema * (1 - _EMA_ALPHA))
        total_val = row["total"] + 1
        correct_val = row["correct"] + (1 if score >= 0.5 else 0)
        new_review_count = (row["review_count"] or 0) + 1

        now = datetime.now()
        next_review_date = scheduler.calculate_next_review_date(new_ema, new_review_count, now)
        review_interval_days = (next_review_date - now).days

        cur.execute(
            """
            UPDATE knowledge_profile
            SET correct = ?, total = ?, ema_score = ?, last_reviewed_at = ?,
                review_interval_days = ?, review_count = ?, next_review_date = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (
                correct_val, total_val, new_ema, now,
                review_interval_days, new_review_count, next_review_date,
                topic_id, user_id
            )
        )
        conn.commit()

    return get_topic_by_id(topic_id, user_id)


def build_review_queue(user_id: int, category: str = "all", limit: int = 10, sort: str = "urgent") -> dict:
    """Return topics due for review today or earlier, filtered and sorted.

    category filters by mastery_category ('weak'/'average'/'strong'), sort picks
    between soonest-due-first ('urgent') and most-recently-studied-first ('recent').
    """
    topics = list_topics_with_schedule(user_id)
    due_topics = [t for t in topics if t["is_due"]]

    if category != "all":
        due_topics = [t for t in due_topics if t["mastery_category"].lower() == category]

    if sort == "urgent":
        due_topics.sort(key=lambda t: t["days_until_review"])
    else:
        due_topics.sort(key=lambda t: t["last_reviewed"] or "", reverse=True)

    overdue_count = sum(1 for t in topics if t["days_until_review"] < 0)

    queue = [
        {
            "id": t["id"],
            "topic": t["topic"],
            "mastery_level": t["mastery_level"],
            "mastery_category": t["mastery_category"],
            "days_until_review": t["days_until_review"],
            "last_reviewed": t["last_reviewed"],
            "next_review": t["next_review"],
            "review_count": t["review_count"],
            "urgency_score": t["urgency_score"],
        }
        for t in due_topics[:limit]
    ]

    return {
        "queue": queue,
        "total_topics": len(topics),
        "overdue_count": overdue_count,
    }


def get_topic_statistics(user_id: int) -> dict:
    """Return dashboard-level stats summarizing a user's spaced-repetition progress."""
    topics = list_topics_with_schedule(user_id)
    if not topics:
        return {
            "total_topics": 0,
            "topics_due": 0,
            "topics_overdue": 0,
            "avg_mastery": 0.0,
            "strongest_topic": None,
            "strongest_mastery": None,
            "weakest_topic": None,
            "weakest_mastery": None,
        }

    topics_due = sum(1 for t in topics if t["is_due"])
    topics_overdue = sum(1 for t in topics if t["days_until_review"] < 0)
    avg_mastery = round(sum(t["mastery_level"] for t in topics) / len(topics), 3)
    strongest = max(topics, key=lambda t: t["mastery_level"])
    weakest = min(topics, key=lambda t: t["mastery_level"])

    return {
        "total_topics": len(topics),
        "topics_due": topics_due,
        "topics_overdue": topics_overdue,
        "avg_mastery": avg_mastery,
        "strongest_topic": strongest["topic"],
        "strongest_mastery": strongest["mastery_level"],
        "weakest_topic": weakest["topic"],
        "weakest_mastery": weakest["mastery_level"],
    }


def get_user_stats(user_id: int) -> dict:
    """Aggregate totals for the user dashboard.

    total_questions: quiz questions answered across all sessions.
    average_mastery_pct: mean EMA score (0-100) across all unique topics.
    """
    global_profile = load_global_profile(user_id)
    total_questions = 0
    all_ema_scores = []

    for subj_topics in global_profile.values():
        for stats in subj_topics.values():
            total_questions += stats.get("total", 0)
            score = stats.get("ema_score")
            if score is not None:
                all_ema_scores.append(score)

    avg_mastery = round((sum(all_ema_scores) / len(all_ema_scores)) * 100, 1) if all_ema_scores else 0.0

    return {
        "total_questions": total_questions,
        "average_mastery_pct": avg_mastery,
    }


def get_review_schedule(topic: str, user_id: int) -> dict:
    """Return the spaced-repetition review schedule for a topic across all subjects.

    Backward compatible: topics that predate spaced repetition (no reviews yet)
    are reported as due today with a review_count of 0.
    """
    topic_key = topic.strip().title()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT subject, topic, ema_score, correct, total, last_reviewed_at,
                   review_interval_days, review_count, next_review_date
            FROM knowledge_profile
            WHERE user_id = ? AND topic = ?
            """,
            (user_id, topic_key)
        )
        row = cur.fetchone()

    if not row:
        return {}

    ema_score = row["ema_score"]
    mastery_level = ema_score if ema_score is not None else (
        row["correct"] / row["total"] if row["total"] else 0.0
    )
    last_reviewed_at = _parse_ts(row["last_reviewed_at"])
    next_review_date = _parse_ts(row["next_review_date"])
    days_until_review = (next_review_date.date() - datetime.now().date()).days if next_review_date else 0

    return {
        "topic": row["topic"],
        "mastery_level": mastery_level,
        "mastery_category": scheduler.get_mastery_category(mastery_level),
        "last_reviewed": last_reviewed_at.date().isoformat() if last_reviewed_at else None,
        "next_review": next_review_date.date().isoformat() if next_review_date else None,
        "days_until_review": days_until_review,
        "review_count": row["review_count"] or 0,
    }
