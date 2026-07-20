"""The student's global knowledge profile (mastery per subject/topic)."""
from fastapi import APIRouter, Depends

from src.api.deps import get_current_user
from src.personalization import mastery

router = APIRouter(prefix="/api/profile", tags=["profile"])


@router.get("")
def get_profile(user_id: int = Depends(get_current_user)):
    """Return tracked topics grouped into weak / average / strong per subject."""
    return mastery.get_performance_areas(user_id)


@router.delete("/{subject}/{topic}")
def delete_profile_topic(subject: str, topic: str, user_id: int = Depends(get_current_user)):
    """Permanently remove a topic from the student's global knowledge profile."""
    mastery.delete_topic(user_id, subject, topic)
    return {"status": "deleted", "subject": subject, "topic": topic}
