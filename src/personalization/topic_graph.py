"""Curriculum graph: subject-scoped topic relationships (prerequisites, related
topics), stored in topic_edges.

This is shared across users (one graph per subject, not per user) — per-user
mastery data stays in knowledge_profile (see mastery.py). Topic names here are
plain strings, normalized the same way mastery.py normalizes them
(.strip().title()), since there's no separate topics dimension table either
side can key off of.
"""
from typing import Dict, List

from src.core.database import get_db

# Guards recursive traversal against a cycle — edges aren't DAG-enforced at the
# database level, so without this a bad edge set (manual or LLM-seeded) could
# recurse forever.
_MAX_CHAIN_DEPTH = 20


def _normalize(value: str) -> str:
    """Match the (subject, topic) normalization mastery.py uses, so graph nodes
    and knowledge_profile rows refer to the same strings."""
    return value.strip().title()


def add_edge(subject: str, from_topic: str, to_topic: str,
             relation: str = "prerequisite_of", weight: float = 1.0,
             source: str = "manual") -> None:
    """Insert a directed edge between two topics in a subject's graph.

    No-ops if the same (subject, from_topic, to_topic, relation) edge already
    exists.
    """
    subj = _normalize(subject)
    f = _normalize(from_topic)
    t = _normalize(to_topic)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO topic_edges (subject, from_topic, to_topic, relation, weight, source)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (subject, from_topic, to_topic, relation) DO NOTHING
            """,
            (subj, f, t, relation, weight, source)
        )
        conn.commit()


def delete_edge(subject: str, from_topic: str, to_topic: str,
                 relation: str = "prerequisite_of") -> None:
    """Remove a single edge."""
    subj = _normalize(subject)
    f = _normalize(from_topic)
    t = _normalize(to_topic)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM topic_edges WHERE subject = %s AND from_topic = %s "
            "AND to_topic = %s AND relation = %s",
            (subj, f, t, relation)
        )
        conn.commit()


def has_edges(subject: str) -> bool:
    """Whether a subject's graph has been seeded yet at all."""
    subj = _normalize(subject)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM topic_edges WHERE subject = %s LIMIT 1", (subj,))
        return cur.fetchone() is not None


def get_prerequisites(subject: str, topic: str) -> List[str]:
    """Direct (one-hop) prerequisites of a topic — topics that must come first."""
    subj = _normalize(subject)
    t = _normalize(topic)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT from_topic FROM topic_edges "
            "WHERE subject = %s AND to_topic = %s AND relation = 'prerequisite_of'",
            (subj, t)
        )
        return [r["from_topic"] for r in cur.fetchall()]


def get_dependents(subject: str, topic: str) -> List[str]:
    """Direct (one-hop) topics that list this one as a prerequisite."""
    subj = _normalize(subject)
    t = _normalize(topic)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT to_topic FROM topic_edges "
            "WHERE subject = %s AND from_topic = %s AND relation = 'prerequisite_of'",
            (subj, t)
        )
        return [r["to_topic"] for r in cur.fetchall()]


def get_prerequisite_chain(subject: str, topic: str) -> List[str]:
    """Full upstream prerequisite chain (all depths) for a topic, nearest-first.

    Depth-capped to guard against cycles, since edges aren't enforced to form
    a DAG at the database level.
    """
    subj = _normalize(subject)
    t = _normalize(topic)
    sql = """
        WITH RECURSIVE chain AS (
            SELECT from_topic, 1 AS depth
            FROM topic_edges
            WHERE subject = %s AND to_topic = %s AND relation = 'prerequisite_of'
            UNION ALL
            SELECT te.from_topic, c.depth + 1
            FROM topic_edges te
            JOIN chain c ON te.to_topic = c.from_topic
            WHERE te.subject = %s AND te.relation = 'prerequisite_of'
              AND c.depth < %s
        )
        SELECT from_topic, MIN(depth) AS depth
        FROM chain
        GROUP BY from_topic
        ORDER BY depth
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (subj, t, subj, _MAX_CHAIN_DEPTH))
        return [r["from_topic"] for r in cur.fetchall()]


def get_subject_node_names(subject: str) -> List[str]:
    """All distinct topic names that appear anywhere in a subject's graph.

    Used to widen the fuzzy-match vocabulary when a new quiz-graded topic
    string needs to be reconciled against the graph's canonical names.
    """
    subj = _normalize(subject)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT from_topic AS topic FROM topic_edges WHERE subject = %s "
            "UNION SELECT to_topic AS topic FROM topic_edges WHERE subject = %s",
            (subj, subj)
        )
        return [r["topic"] for r in cur.fetchall()]


def get_subject_graph(subject: str) -> Dict:
    """All nodes/edges for a subject, shaped for a graph visualization."""
    subj = _normalize(subject)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT from_topic, to_topic, relation, weight FROM topic_edges WHERE subject = %s",
            (subj,)
        )
        edges = cur.fetchall()

    nodes = sorted({e["from_topic"] for e in edges} | {e["to_topic"] for e in edges})
    return {
        "subject": subj,
        "nodes": nodes,
        "edges": [
            {
                "from": e["from_topic"],
                "to": e["to_topic"],
                "relation": e["relation"],
                "weight": e["weight"],
            }
            for e in edges
        ],
    }


def prioritize_prerequisites(due_topics: List[dict]) -> List[dict]:
    """Reorder a due-topics list (from mastery.list_topics_with_schedule) so a
    topic's still-weak prerequisites are bumped ahead of it.

    Stable otherwise — only reorders when a prerequisite relationship actually
    exists between two topics already in the list, so it doesn't fight the
    urgent/recent ordering build_review_queue already applied.
    """
    if not due_topics:
        return due_topics

    by_key = {(t["subject"], t["topic"]): t for t in due_topics}
    reordered: List[dict] = []
    placed = set()

    for t in due_topics:
        key = (t["subject"], t["topic"])
        if key in placed:
            continue
        if t["mastery_category"].lower() == "weak":
            for prereq in get_prerequisites(t["subject"], t["topic"]):
                prereq_key = (t["subject"], prereq)
                prereq_topic = by_key.get(prereq_key)
                if (prereq_topic is not None
                        and prereq_key not in placed
                        and prereq_topic["mastery_category"].lower() == "weak"):
                    reordered.append(prereq_topic)
                    placed.add(prereq_key)
        reordered.append(t)
        placed.add(key)

    return reordered
