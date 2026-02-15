from __future__ import annotations

import logging
import os
import random
import time
from collections import deque
from typing import Any, Dict, Optional

from ..dynamo.preprints_repo import PreprintsRepo
from .data_assembly import assemble_email_context
from .gmail import send_email
from .suppression import is_suppressed
from .template import render_email
from .validation import validate_recipient

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Simple sliding-window rate limiter tracking send timestamps in-memory."""

    def __init__(self):
        self.per_minute = int(os.environ.get("EMAIL_RATE_PER_MINUTE", "10"))
        self.per_hour = int(os.environ.get("EMAIL_RATE_PER_HOUR", "100"))
        self.per_day = int(os.environ.get("EMAIL_RATE_PER_DAY", "500"))
        self._timestamps: deque[float] = deque()

    def _prune(self, now: float) -> None:
        day_ago = now - 86400
        while self._timestamps and self._timestamps[0] < day_ago:
            self._timestamps.popleft()

    def _count_since(self, now: float, seconds: float) -> int:
        cutoff = now - seconds
        return sum(1 for ts in self._timestamps if ts >= cutoff)

    def wait_if_needed(self) -> bool:
        """Wait until sending is allowed. Returns False if daily limit reached."""
        while True:
            now = time.time()
            self._prune(now)

            day_count = len(self._timestamps)
            if day_count >= self.per_day:
                return False

            hour_count = self._count_since(now, 3600)
            if hour_count >= self.per_hour:
                sleep_time = self._timestamps[-self.per_hour] + 3600 - now + 0.1
                logger.info("Rate limit (hourly): sleeping %.1fs", sleep_time)
                time.sleep(max(sleep_time, 0.1))
                continue

            minute_count = self._count_since(now, 60)
            if minute_count >= self.per_minute:
                sleep_time = self._timestamps[-self.per_minute] + 60 - now + 0.1
                logger.info("Rate limit (per-minute): sleeping %.1fs", sleep_time)
                time.sleep(max(sleep_time, 0.1))
                continue

            return True

    def record_send(self) -> None:
        self._timestamps.append(time.time())


def process_email_batch(
    *,
    limit: int = 50,
    max_seconds: Optional[int] = None,
    spread_seconds: Optional[int] = None,
    dry_run: bool = False,
    osf_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a batch of eligible preprints for email sending.

    Each preprint maps to exactly one recipient, so *limit* caps the
    number of recipients contacted.

    When *spread_seconds* is set the batch sleeps a randomised interval
    between sends so that emails are spread over roughly that many
    seconds rather than fired in a burst.

    Returns a summary dict with counts of sent, failed, skipped, etc.
    """
    repo = PreprintsRepo()
    rate_limiter = _RateLimiter()

    deadline = (time.monotonic() + max_seconds) if max_seconds and max_seconds > 0 else None

    # Select eligible preprints
    if osf_id:
        item = repo.t_preprints.get_item(Key={"osf_id": osf_id}).get("Item")
        candidates = [item] if item else []
    else:
        candidates = repo.select_for_email(limit=limit)

    # Pre-compute average inter-send delay when spreading is requested.
    avg_delay = 0.0
    if spread_seconds and spread_seconds > 0 and len(candidates) > 1:
        avg_delay = spread_seconds / len(candidates)

    sent = 0
    failed = 0
    skipped_suppressed = 0
    skipped_invalid = 0
    skipped_no_context = 0

    for preprint in candidates:
        if deadline and time.monotonic() >= deadline:
            break
        if sent + failed >= limit:
            break

        pid = preprint.get("osf_id")
        if not pid:
            continue

        # Assemble context
        context = assemble_email_context(pid, repo=repo)
        if not context:
            skipped_no_context += 1
            continue

        email_addr = context.get("_email_address", "")

        # Check suppression
        if is_suppressed(email_addr, repo=repo):
            skipped_suppressed += 1
            logger.info("Skipped suppressed email", extra={"osf_id": pid, "email": email_addr})
            continue

        # Validate email
        valid, err_msg = validate_recipient(email_addr)
        repo.mark_email_validated(pid, "valid" if valid else f"invalid: {err_msg}")
        if not valid:
            skipped_invalid += 1
            logger.info("Skipped invalid email", extra={"osf_id": pid, "email": email_addr, "error": err_msg})
            continue

        # Render email
        try:
            subject, html_body, plain_body = render_email(context)
        except Exception as exc:
            failed += 1
            repo.mark_email_error(pid, f"template render: {exc}")
            logger.exception("Template render failed", extra={"osf_id": pid})
            continue

        if dry_run:
            sent += 1
            logger.info(
                "DRY RUN: would send email",
                extra={"osf_id": pid, "to": email_addr, "subject": subject},
            )
            continue

        # Random inter-send delay to spread emails over time
        if avg_delay > 0 and sent > 0:
            delay = random.uniform(0.5 * avg_delay, 1.5 * avg_delay)
            remaining = (deadline - time.monotonic()) if deadline else float("inf")
            delay = min(delay, max(remaining - 5, 0))  # keep 5s headroom
            if delay > 0:
                logger.info("Spread delay: sleeping %.1fs", delay)
                time.sleep(delay)
                if deadline and time.monotonic() >= deadline:
                    break

        # Rate limiting
        if not rate_limiter.wait_if_needed():
            logger.warning("Daily rate limit reached, stopping batch")
            break

        # Send
        try:
            result = send_email(to=email_addr, subject=subject, html_body=html_body, plain_body=plain_body)
            message_id = result.get("id", "")
            repo.mark_email_sent(pid, recipient=email_addr, message_id=message_id)
            rate_limiter.record_send()
            sent += 1
        except Exception as exc:
            failed += 1
            repo.mark_email_error(pid, str(exc))
            logger.exception("Email send failed", extra={"osf_id": pid, "to": email_addr})

    return {
        "stage": "email",
        "sent": sent,
        "failed": failed,
        "skipped_suppressed": skipped_suppressed,
        "skipped_invalid": skipped_invalid,
        "skipped_no_context": skipped_no_context,
        "dry_run": dry_run,
        "stopped_due_to_time": bool(deadline and time.monotonic() >= deadline),
    }
