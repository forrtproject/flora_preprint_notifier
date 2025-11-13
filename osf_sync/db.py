from __future__ import annotations

from .dynamo.tables import ensure_tables
from .dynamo.client import get_dynamo_resource


def init_db() -> None:
    """
    Initialize DynamoDB tables (no-op if they already exist).

    Docker Compose and tasks import this as a stable entrypoint.
    """
    ensure_tables()


__all__ = ["init_db", "get_dynamo_resource"]

