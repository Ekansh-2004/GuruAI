"""Pydantic request models for the HTTP API."""
from typing import Optional

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    username: str
    password: str
    name: Optional[str] = "The Scholar"


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    session_id: str
    question: str


class QuizAnswerRequest(BaseModel):
    session_id: str
    subject: str
    topic: str
    is_correct: bool


class MarkReviewedRequest(BaseModel):
    score: int
    notes: Optional[str] = None


class SubjectRequest(BaseModel):
    subject: str


class TopicExplainRequest(BaseModel):
    topic: str
    subject: str
    mastery_level: str  # "strong", "average", "weak"
    score_pct: int      # 0-100


class MemoryMessageRequest(BaseModel):
    message: str


class UserProfileRequest(BaseModel):
    name: str
    bio: str
