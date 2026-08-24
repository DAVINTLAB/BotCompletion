"""Date utilities for account age calculations."""

import datetime
from typing import Optional


def calculate_account_age(
    created_at: str,
    reference_date: datetime.datetime,
    date_format: str,
    min_age_days: int = 3
) -> Optional[int]:
    """
    Calculate the account age in days from creation date to reference date.

    Args:
        created_at: The account creation date string
        reference_date: The reference date to calculate age against (timezone-aware)
        date_format: The strptime format string for parsing created_at
        min_age_days: Minimum age to return if account is newer than reference date

    Returns:
        Number of days since account creation (minimum min_age_days if account
        is newer than reference date), or None if parsing fails
    """
    if not created_at:
        return None

    try:
        created_at = created_at.strip()
        try:
            created_date = datetime.datetime.strptime(created_at, date_format)
        except ValueError:
            # Twibot-22 timestamps include a timezone offset like "+00:00" that
            # %Y-%m-%d %H:%M:%S can't parse. Fall back to fromisoformat which
            # handles ISO-8601 strings with offsets natively.
            created_date = datetime.datetime.fromisoformat(created_at)

        # Ensure both dates are timezone-aware for comparison
        if created_date.tzinfo is None:
            created_date = created_date.replace(tzinfo=datetime.timezone.utc)

        total_days = (reference_date - created_date).days

        # If account is newer than reference date, return minimum age
        if total_days < 0:
            return min_age_days

        return total_days
    except (ValueError, TypeError):
        return None


def format_account_age(total_days: Optional[int]) -> Optional[str]:
    """
    Format account age in days to a human-readable string.

    Args:
        total_days: Number of days since account creation

    Returns:
        Formatted string like "X years, Y days" or "Y days", or None if input is None
    """
    if total_days is None:
        return None

    years = total_days // 365
    days = total_days % 365

    if years == 0:
        return f"{days} day{'s' if days != 1 else ''}"
    else:
        return f"{years} year{'s' if years != 1 else ''}, {days} day{'s' if days != 1 else ''}"
