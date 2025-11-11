from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import os
from .dynamo import table
from .db import PREPRINTS_T

def _as_iso_date(s: Optional[str]) -> Optional[str]:
    # OSF gives full timestamps. We keep YYYY-MM-DD for GSI partitioning.
    if not s:
        return None
    try:
        if "T" in s:
            return s.split("T", 1)[0]
        return s[:10]
    except Exception:
        return None

def _newer(existing: Dict, incoming: Dict) -> bool:
    """
    Emulate your previous 'only update if newer' logic.
    Compare 'date_modified' and 'version' when present.
    """
    ex_mod = existing.get("date_modified")
    in_mod = incoming.get("date_modified")
    ex_ver = existing.get("version") or 0
    in_ver = incoming.get("version") or 0

    try:
        if ex_mod and in_mod:
            ex_dt = datetime.fromisoformat(ex_mod.replace("Z", "+00:00"))
            in_dt = datetime.fromisoformat(in_mod.replace("Z", "+00:00"))
            if in_dt > ex_dt:
                return True
        elif in_mod and not ex_mod:
            return True
    except Exception:
        pass

    return in_ver > ex_ver

def upsert_batch(batch: List[Dict[str, Any]]) -> int:
    """
    Accepts a list of OSF API objects. Insert/update into DynamoDB `preprints`.
    """
    t = table(PREPRINTS_T)
    n = 0
    for obj in batch:
        attrs = obj.get("attributes") or {}
        osf_id = obj.get("id")
        if not osf_id:
            continue

        # Prepare item similar to your SQL schema
        item = {
            "osf_id": osf_id,
            "type": obj.get("type"),
            "title": attrs.get("title"),
            "description": attrs.get("description"),
            "doi": attrs.get("doi"),
            "date_created": attrs.get("date_created"),
            "date_modified": attrs.get("date_modified"),
            "date_published": attrs.get("date_published"),
            "is_published": bool(attrs.get("is_published")),
            "version": attrs.get("version"),
            "is_latest_version": bool(attrs.get("is_latest_version")),
            "reviews_state": attrs.get("reviews_state"),
            "tags": attrs.get("tags") or [],
            "subjects": attrs.get("subjects") or [],
            "license_record": attrs.get("license_record"),
            "provider_id": (obj.get("relationships") or {}).get("provider", {}).get("data", {}).get("id"),
            "links": obj.get("links") or {},
            "raw": obj,  # keep raw for downstream
            # denormalized flags for GSIs:
            "pdf_downloaded": False,
            "tei_generated": False,
            "tei_extracted": False,
            "_updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Normalize for GSI
        iso_pub = _as_iso_date(item["date_published"])
        if iso_pub:
            item["date_published"] = iso_pub

        # Read existing and decide
        existing = t.get_item(Key={"osf_id": osf_id}).get("Item")
        if existing:
            if not _newer(existing, item):
                continue
            # merge booleans (keep any true)
            for f in ("pdf_downloaded", "tei_generated", "tei_extracted"):
                if existing.get(f):
                    item[f] = True

        t.put_item(Item=item)
        n += 1
    return n