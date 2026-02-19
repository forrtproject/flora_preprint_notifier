from __future__ import annotations

import logging

from .blacklist import is_blacklisted_email

log = logging.getLogger(__name__)


def is_suppressed(email: str, repo=None) -> bool:
    """Check if an email address is suppressed.

    Checks the file-based blacklist first (fast, in-memory LRU cache),
    then checks the DynamoDB suppression table for personal addresses
    (bounces, unsubscribes).

    Accepts an optional ``SuppressionRepo`` instance to avoid
    re-creating one per call.  Falls back to a lazy import if not passed.
    Fails open on DynamoDB errors so transient DB issues don't block sending.
    """
    if not email or not email.strip():
        return True

    # Fast path: file-based blacklist (generic patterns)
    if is_blacklisted_email(email):
        return True

    # DynamoDB suppression table (personal addresses)
    try:
        if repo is None:
            from ..dynamo.suppression_repo import SuppressionRepo
            repo = SuppressionRepo()
        return repo.is_suppressed(email)
    except Exception:
        log.warning("DynamoDB suppression check failed (fail-open)", exc_info=True)
        return False


def add_suppression(email: str, reason: str, repo=None) -> None:
    """Add an email address to the DynamoDB suppression table."""
    try:
        if repo is None:
            from ..dynamo.suppression_repo import SuppressionRepo
            repo = SuppressionRepo()
        repo.add_suppression(email, reason)
    except Exception:
        log.warning("DynamoDB suppression add failed", exc_info=True)
