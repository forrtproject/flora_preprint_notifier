from __future__ import annotations

import calendar
import datetime as dt
import os
from typing import Any, Dict, Optional

from .client import get_dynamo_resource

CACHE_TABLE = os.environ.get("DDB_TABLE_API_CACHE", "api_cache")
CACHE_TTL_MONTHS_DEFAULT = int(os.environ.get("API_CACHE_TTL_MONTHS", "6"))


def _strip_nones(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of dict without None values (DynamoDB rejects None)."""
    return {k: v for k, v in d.items() if v is not None}


def _add_months(ts: dt.datetime, months: int) -> dt.datetime:
    year = ts.year
    month = ts.month + months
    while month > 12:
        month -= 12
        year += 1
    day = min(ts.day, calendar.monthrange(year, month)[1])
    return ts.replace(year=year, month=month, day=day)


def _expires_at_epoch(ttl_months: int) -> int:
    now = dt.datetime.now(dt.timezone.utc)
    exp = _add_months(now, ttl_months)
    return int(exp.timestamp())


def _expires_at_epoch_seconds(ttl_seconds: int) -> int:
    now = dt.datetime.now(dt.timezone.utc)
    exp = now + dt.timedelta(seconds=ttl_seconds)
    return int(exp.timestamp())


def _parse_iso_dt(value: Optional[str]) -> Optional[dt.datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        ts = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    else:
        ts = ts.astimezone(dt.timezone.utc)
    return ts


class ApiCacheRepo:
    """
    Simple cache table helper. Items are keyed by cache_key and expire via DynamoDB TTL.
    """
    def __init__(self) -> None:
        ddb = get_dynamo_resource()
        self.t_cache = ddb.Table(CACHE_TABLE)

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        resp = self.t_cache.get_item(Key={"cache_key": cache_key})
        return resp.get("Item")

    def put(
        self,
        cache_key: str,
        payload: Any,
        *,
        source: Optional[str] = None,
        ttl_months: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
        status: Optional[bool] = None,
        checked_at: Optional[str] = None,
    ) -> None:
        now_dt = dt.datetime.now(dt.timezone.utc)
        now = now_dt.isoformat()
        item = {
            "cache_key": cache_key,
            "source": source,
            "payload": payload,
            "status": status,
            "cached_at": now,
            "checked_at": checked_at or now,
            "expires_at": _expires_at_epoch_seconds(int(ttl_seconds)) if ttl_seconds is not None
            else _expires_at_epoch(CACHE_TTL_MONTHS_DEFAULT if ttl_months is None else int(ttl_months)),
        }
        self.t_cache.put_item(Item=_strip_nones(item))

    def is_fresh(self, item: Optional[Dict[str, Any]], *, ttl_seconds: Optional[int] = None) -> bool:
        if not item:
            return False
        now = dt.datetime.now(dt.timezone.utc)
        exp = item.get("expires_at")
        try:
            if exp is not None and int(exp) > int(now.timestamp()):
                return True
        except Exception:
            pass
        if ttl_seconds is None:
            return False
        cached_at = _parse_iso_dt(item.get("cached_at"))
        if not cached_at:
            return False
        age = (now - cached_at).total_seconds()
        return age <= ttl_seconds
