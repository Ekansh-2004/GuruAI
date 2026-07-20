"""The subjects a user has registered for study."""
from fastapi import APIRouter, Depends

from src.api.deps import get_current_user
from src.api.schemas import SubjectRequest
from src.personalization import user_memory

router = APIRouter(prefix="/api/subjects", tags=["subjects"])


@router.get("")
def get_subjects(user_id: int = Depends(get_current_user)):
    """Return all registered subjects."""
    return {"subjects": user_memory.load_subjects(user_id)}


@router.post("")
def add_subject(req: SubjectRequest, user_id: int = Depends(get_current_user)):
    """Add a new subject to the user's study profile."""
    return {"subjects": user_memory.save_subject(user_id, req.subject)}


@router.delete("/{subject}")
def remove_subject(subject: str, user_id: int = Depends(get_current_user)):
    """Remove a subject from the user's study profile."""
    return {"subjects": user_memory.delete_subject(user_id, subject)}
