from __future__ import annotations

import datetime as dt
import logging

from ..dynamo.preprints_repo import PreprintsRepo

logger = logging.getLogger(__name__)


def is_suppressed(email: str, repo: PreprintsRepo | None = None) -> bool:
    """Check if an email address is on the suppression list.

    Fail-safe: if the lookup fails, treat as suppressed (don't send if unsure).
    """
    if not email or not email.strip():
        return True
    repo = repo or PreprintsRepo()
    try:
        item = repo.t_suppression.get_item(Key={"email": email.lower().strip()}).get("Item")
        return bool(item)
    except Exception:
        logger.exception("Suppression check failed; treating as suppressed", extra={"email": email})
        return True


def add_suppression(email: str, reason: str, repo: PreprintsRepo | None = None) -> None:
    """Add an email address to the suppression list."""
    repo = repo or PreprintsRepo()
    now = dt.datetime.utcnow().isoformat()
    repo.t_suppression.put_item(Item={
        "email": email.lower().strip(),
        "reason": reason,
        "suppressed_at": now,
    })
