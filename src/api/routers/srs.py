"""Spaced-repetition review queue and study-session recording."""
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_current_user
from src.api.schemas import MarkReviewedRequest
from src.personalization import mastery

router = APIRouter(prefix="/api", tags=["srs"])


@router.get("/suggestions/review-queue")
def get_review_queue(
    category: str = Query("all", pattern="^(all|weak|average|strong)$"),
    limit: int = Query(10, ge=1, le=50),
    sort: str = Query("urgent", pattern="^(urgent|recent)$"),
    user_id: int = Depends(get_current_user),
) -> dict:
    """Return topics due for review today or earlier.

    category filters by mastery_category ('weak'/'average'/'strong'), sort picks
    between soonest-due-first ('urgent') and most-recently-studied-first ('recent').
    """
    return mastery.build_review_queue(user_id, category=category, limit=limit, sort=sort)


@router.post("/topics/{topic_id}/mark-reviewed")
def mark_topic_reviewed(
    topic_id: int,
    req: MarkReviewedRequest,
    user_id: int = Depends(get_current_user),
) -> dict:
    """Record a study session for a topic: updates its mastery EMA and spaced-repetition schedule.

    score is 0-10. notes is accepted for future use but is not currently persisted
    (knowledge_profile has no notes column).
    """
    if not (0 <= req.score <= 10):
        raise HTTPException(status_code=400, detail="score must be between 0 and 10")

    updated = mastery.update_ema(topic_id, user_id, req.score / 10)
    if not updated:
        raise HTTPException(status_code=404, detail="Topic not found")

    return {
        "topic": updated["topic"],
        "mastery_updated": updated["mastery_level"],
        "next_review": updated["next_review"],
        "message": "Great! Study session recorded.",
    }


@router.get("/topics/statistics")
def get_topics_statistics(user_id: int = Depends(get_current_user)) -> dict:
    """Return dashboard stats summarizing the user's spaced-repetition progress."""
    return mastery.get_topic_statistics(user_id)
