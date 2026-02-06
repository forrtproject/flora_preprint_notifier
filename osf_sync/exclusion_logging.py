from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .dynamo.preprints_repo import PreprintsRepo

logger = logging.getLogger(__name__)


def log_preprint_exclusion(
    *,
    reason: str,
    osf_id: Optional[str] = None,
    stage: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Best-effort exclusion registry write (one row per excluded osf_id).
    Any failure is logged and ignored so pipeline flow never depends on
    reporting writes.
    """
    try:
        if not osf_id:
            logger.warning("cannot record exclusion without osf_id", extra={"reason": reason, "stage": stage})
            return False
        repo = PreprintsRepo()
        return repo.mark_preprint_excluded(osf_id=osf_id, reason=reason, stage=stage, details=details)
    except Exception:
        logger.warning(
            "failed to record preprint exclusion",
            exc_info=True,
            extra={"reason": reason, "osf_id": osf_id, "stage": stage},
        )
        return False
