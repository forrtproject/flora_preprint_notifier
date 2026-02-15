from __future__ import annotations

import logging
from typing import Optional, Tuple

from email_validator import validate_email, EmailNotValidError

logger = logging.getLogger(__name__)


def validate_recipient(email: str) -> Tuple[bool, Optional[str]]:
    """Validate an email address for syntax and MX deliverability.

    Returns (True, None) on success, or (False, error_message) on failure.
    """
    try:
        result = validate_email(email, check_deliverability=True)
        # Use the normalized form
        _ = result.normalized
        return True, None
    except EmailNotValidError as e:
        return False, str(e)
