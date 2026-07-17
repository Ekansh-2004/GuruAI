"""Spaced-repetition scheduling for topic mastery review.

Schedules the next review date for a topic using exponential backoff:
the base interval depends on how well the topic is mastered, and it
widens further with each successful review.
"""
from datetime import datetime, timedelta
from typing import Optional

WEAK_THRESHOLD = 0.3
STRONG_THRESHOLD = 0.7

_BASE_INTERVALS = {
    "Weak": 1,
    "Average": 3,
    "Strong": 7,
}


class SpacedRepetitionScheduler:
    """Computes review scheduling for a topic based on mastery level.

    Example:
        >>> scheduler = SpacedRepetitionScheduler()
        >>> scheduler.get_mastery_category(0.2)
        'Weak'
        >>> next_date = scheduler.calculate_next_review_date(
        ...     mastery_level=0.8, review_count=1, last_reviewed_at=datetime(2024, 7, 15)
        ... )
        >>> next_date
        datetime.datetime(2024, 7, 29, 0, 0)
    """

    def __init__(self, base_interval_days: int = 1, max_interval_days: int = 60) -> None:
        """Initialize scheduler with configurable intervals.

        Args:
            base_interval_days: Fallback interval (days) used if no category-specific
                base interval applies. Currently unused by the built-in Weak/Average/
                Strong bands but kept for callers that want a custom floor.
            max_interval_days: Hard cap on the computed review interval, in days.
        """
        self.base_interval_days = base_interval_days
        self.max_interval_days = max_interval_days

    def get_mastery_category(self, mastery_level: float) -> str:
        """Return 'Weak' | 'Average' | 'Strong' based on mastery level.

        Example:
            >>> SpacedRepetitionScheduler().get_mastery_category(0.5)
            'Average'
        """
        if mastery_level < WEAK_THRESHOLD:
            return "Weak"
        if mastery_level < STRONG_THRESHOLD:
            return "Average"
        return "Strong"

    def calculate_next_review_date(
        self, mastery_level: float, review_count: int, last_reviewed_at: datetime
    ) -> datetime:
        """Calculate the next review date using exponential backoff based on mastery.

        The category base interval (Weak=1, Average=3, Strong=7 days) is multiplied
        by (review_count + 1) to widen the gap after each successful review, then
        capped at max_interval_days.

        Example:
            >>> s = SpacedRepetitionScheduler()
            >>> s.calculate_next_review_date(0.1, 0, datetime(2024, 7, 15))
            datetime.datetime(2024, 7, 16, 0, 0)
            >>> s.calculate_next_review_date(0.1, 2, datetime(2024, 7, 15))
            datetime.datetime(2024, 7, 18, 0, 0)
        """
        category = self.get_mastery_category(mastery_level)
        base_days = _BASE_INTERVALS[category]
        interval_days = min(base_days * (review_count + 1), self.max_interval_days)
        return last_reviewed_at + timedelta(days=interval_days)

    def is_due_for_review(
        self, last_reviewed_at: Optional[datetime], next_review_date: Optional[datetime]
    ) -> bool:
        """Check if a topic is due for review today or earlier.

        A topic that has never been reviewed, or has no scheduled next review,
        is always considered due.

        Example:
            >>> s = SpacedRepetitionScheduler()
            >>> s.is_due_for_review(None, None)
            True
        """
        if last_reviewed_at is None or next_review_date is None:
            return True
        return next_review_date.date() <= datetime.now().date()
