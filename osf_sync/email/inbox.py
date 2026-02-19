"""Scan Gmail inbox via IMAP for bounces and unsubscribe replies.

Detected addresses are stored in the DynamoDB suppression table so they
are excluded from future email sends.
"""
from __future__ import annotations

import email
import email.policy
import imaplib
import logging
import os
import re
from typing import Any, Dict, Optional, Set

log = logging.getLogger(__name__)

# RFC 5322 simplified email regex — good enough for bounce/unsub extraction
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def _sender_address() -> str:
    return os.environ.get("GMAIL_SENDER_ADDRESS", "flora@replications.forrt.org")


def _is_prod() -> bool:
    return (os.environ.get("PIPELINE_ENV", "dev") or "dev").strip().lower() == "prod"


def _extract_bounce_addresses(msg: email.message.Message) -> Set[str]:
    """Extract failed recipient addresses from a bounce (DSN) message."""
    addresses: Set[str] = set()
    self_addr = _sender_address().lower()

    # Walk MIME parts looking for message/delivery-status (RFC 3464)
    for part in msg.walk():
        ct = part.get_content_type()
        if ct == "message/delivery-status":
            payload = part.get_payload()
            if isinstance(payload, list):
                for dsn_part in payload:
                    text = str(dsn_part)
                    for line in text.splitlines():
                        line_lower = line.lower().strip()
                        if line_lower.startswith("final-recipient:") or line_lower.startswith("original-recipient:"):
                            found = _EMAIL_RE.findall(line)
                            addresses.update(a.lower() for a in found if a.lower() != self_addr)
            elif isinstance(payload, str):
                for line in payload.splitlines():
                    line_lower = line.lower().strip()
                    if line_lower.startswith("final-recipient:") or line_lower.startswith("original-recipient:"):
                        found = _EMAIL_RE.findall(line)
                        addresses.update(a.lower() for a in found if a.lower() != self_addr)

    # Fallback: scan plain-text body for emails if DSN parsing found nothing
    if not addresses:
        for part in msg.walk():
            ct = part.get_content_type()
            if ct in ("text/plain", "text/html"):
                try:
                    body = part.get_payload(decode=True)
                    if body:
                        text = body.decode("utf-8", errors="replace")
                        found = _EMAIL_RE.findall(text)
                        addresses.update(a.lower() for a in found if a.lower() != self_addr)
                except Exception:
                    continue

    return addresses


def _extract_unsub_sender(msg: email.message.Message) -> Optional[str]:
    """Extract the sender address from an unsubscribe reply."""
    self_addr = _sender_address().lower()
    from_header = msg.get("From", "")
    found = _EMAIL_RE.findall(from_header)
    for addr in found:
        if addr.lower() != self_addr:
            return addr.lower()
    return None


def process_inbox(
    *,
    max_messages: int = 200,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Scan Gmail inbox for bounces and unsubscribe replies.

    Returns a stats dict with counts of what was found and processed.
    """
    from ..dynamo.suppression_repo import SuppressionRepo

    sender = _sender_address()
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not app_password:
        log.warning("GMAIL_APP_PASSWORD not set, skipping inbox processing")
        return {"bounces_found": 0, "unsubscribes_found": 0, "already_suppressed": 0, "errors": 0, "dry_run": dry_run, "skipped": "no credentials"}

    repo = SuppressionRepo()
    is_prod = _is_prod()

    bounces_found = 0
    unsubscribes_found = 0
    already_suppressed = 0
    errors = 0

    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        imap.login(sender, app_password)
        imap.select("INBOX")
    except Exception:
        log.warning("IMAP connection failed", exc_info=True)
        return {"bounces_found": 0, "unsubscribes_found": 0, "already_suppressed": 0, "errors": 1, "dry_run": dry_run}

    try:
        # --- Process bounces ---
        bounce_ids = _search_unseen(imap, 'FROM "mailer-daemon"', max_messages)
        for msg_id in bounce_ids:
            try:
                msg = _fetch_message(imap, msg_id)
                if msg is None:
                    continue
                addrs = _extract_bounce_addresses(msg)
                for addr in addrs:
                    bounces_found += 1
                    if not dry_run:
                        added = repo.add_suppression(addr, "bounce")
                        if not added:
                            already_suppressed += 1
                if addrs and is_prod and not dry_run:
                    imap.store(msg_id, "+FLAGS", "\\Seen")
            except Exception:
                errors += 1
                log.warning("Error processing bounce message", exc_info=True)

        # --- Process unsubscribes ---
        unsub_ids = _search_unseen(imap, 'SUBJECT "Unsubscribe"', max_messages)
        for msg_id in unsub_ids:
            try:
                msg = _fetch_message(imap, msg_id)
                if msg is None:
                    continue
                addr = _extract_unsub_sender(msg)
                if addr:
                    unsubscribes_found += 1
                    if not dry_run:
                        added = repo.add_suppression(addr, "unsubscribe")
                        if not added:
                            already_suppressed += 1
                    if is_prod and not dry_run:
                        imap.store(msg_id, "+FLAGS", "\\Seen")
            except Exception:
                errors += 1
                log.warning("Error processing unsubscribe message", exc_info=True)
    finally:
        try:
            imap.close()
            imap.logout()
        except Exception:
            pass

    return {
        "bounces_found": bounces_found,
        "unsubscribes_found": unsubscribes_found,
        "already_suppressed": already_suppressed,
        "errors": errors,
        "dry_run": dry_run,
    }


def _search_unseen(imap: imaplib.IMAP4_SSL, criteria: str, max_messages: int) -> list:
    """Search for UNSEEN messages matching criteria, return up to max_messages IDs."""
    try:
        status, data = imap.search(None, "UNSEEN", criteria)
        if status != "OK" or not data or not data[0]:
            return []
        ids = data[0].split()
        return ids[:max_messages]
    except Exception:
        log.warning("IMAP search failed", exc_info=True)
        return []


def _fetch_message(imap: imaplib.IMAP4_SSL, msg_id: bytes) -> Optional[email.message.Message]:
    """Fetch and parse a single message by ID."""
    try:
        status, data = imap.fetch(msg_id, "(RFC822)")
        if status != "OK" or not data or not data[0]:
            return None
        raw = data[0][1]
        if isinstance(raw, bytes):
            return email.message_from_bytes(raw, policy=email.policy.default)
        return None
    except Exception:
        log.warning("IMAP fetch failed for message %s", msg_id, exc_info=True)
        return None
