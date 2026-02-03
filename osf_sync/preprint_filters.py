from __future__ import annotations

import datetime as dt
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv


load_dotenv()

MIN_ORIGINAL_PUBLICATION_DATE_ENV = "OSF_MIN_ORIGINAL_PUBLICATION_DATE"


def parse_iso_date(value: Any) -> Optional[dt.date]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    v = str(value).strip()
    if not v:
        return None
    if len(v) == 4 and v.isdigit():
        try:
            return dt.date(int(v), 1, 1)
        except ValueError:
            return None
    if "T" in v:
        v = v.split("T", 1)[0]
    try:
        return dt.date.fromisoformat(v)
    except ValueError:
        return None


def get_min_original_publication_date() -> Optional[dt.date]:
    raw = os.environ.get(MIN_ORIGINAL_PUBLICATION_DATE_ENV, "2024-01-01")
    raw = (raw or "").strip()
    if not raw:
        return None
    return parse_iso_date(raw)


def original_publication_date_from_obj(obj: Dict[str, Any]) -> Optional[dt.date]:
    attrs = (obj.get("attributes") or {})
    value = attrs.get("original_publication_date") or obj.get("original_publication_date")
    return parse_iso_date(value)


def original_publication_date_from_item(item: Dict[str, Any]) -> Optional[dt.date]:
    value = item.get("original_publication_date")
    if value:
        return parse_iso_date(value)
    raw = item.get("raw") or {}
    attrs = (raw.get("attributes") or {})
    return parse_iso_date(attrs.get("original_publication_date"))


def is_preprint_before_min_date(
    obj: Dict[str, Any], min_date: Optional[dt.date] = None
) -> bool:
    if min_date is None:
        min_date = get_min_original_publication_date()
    if not min_date:
        return False
    pub_date = original_publication_date_from_obj(obj)
    return bool(pub_date and pub_date < min_date)


def should_keep_preprint(obj: Dict[str, Any], min_date: Optional[dt.date] = None) -> bool:
    return not is_preprint_before_min_date(obj, min_date=min_date)
