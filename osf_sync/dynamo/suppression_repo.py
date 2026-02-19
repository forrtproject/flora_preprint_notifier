from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from boto3.dynamodb.conditions import Attr

from .client import get_dynamo_resource
from .tables import EMAIL_SUPPRESSION_TABLE

log = logging.getLogger(__name__)


class SuppressionRepo:
    """DynamoDB-backed email suppression list (bounces, unsubscribes)."""

    def __init__(self) -> None:
        ddb = get_dynamo_resource()
        self.table = ddb.Table(EMAIL_SUPPRESSION_TABLE)

    def is_suppressed(self, email: str) -> bool:
        """Return True if the email is in the suppression table."""
        key = (email or "").lower().strip()
        if not key:
            return False
        resp = self.table.get_item(Key={"email": key})
        return "Item" in resp

    def add_suppression(self, email: str, reason: str) -> bool:
        """Add an email to the suppression table.

        Returns True if the item was newly created, False if it already existed.
        Uses a conditional write to avoid overwriting an existing entry.
        """
        key = (email or "").lower().strip()
        if not key:
            return False
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            self.table.put_item(
                Item={"email": key, "reason": reason, "suppressed_at": now},
                ConditionExpression=Attr("email").not_exists(),
            )
            return True
        except self.table.meta.client.exceptions.ConditionalCheckFailedException:
            return False

    def remove_suppression(self, email: str) -> None:
        """Remove an email from the suppression table (admin use for false-positive bounces)."""
        key = (email or "").lower().strip()
        if not key:
            return
        self.table.delete_item(Key={"email": key})

    def list_suppressions(self) -> List[Dict[str, Any]]:
        """Scan the full suppression table (expected to be small)."""
        items: List[Dict[str, Any]] = []
        resp = self.table.scan()
        items.extend(resp.get("Items", []))
        while resp.get("LastEvaluatedKey"):
            resp = self.table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items.extend(resp.get("Items", []))
        return items
