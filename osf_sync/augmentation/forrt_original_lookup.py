from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from ..dynamo.preprints_repo import PreprintsRepo
from ..logging_setup import get_logger, with_extras

logger = get_logger(__name__)

FORRT_ENDPOINT = os.environ.get("FORRT_ORIGINAL_LOOKUP_URL", "https://rep-api.forrt.org/v1/original-lookup")
FORRT_CACHE_PATH_DEFAULT = os.environ.get("FORRT_CACHE_PATH", os.path.join("data", "forrt_lookup_cache.json"))
FORRT_CACHE_TTL_HOURS_DEFAULT = int(os.environ.get("FORRT_CACHE_TTL_HOURS", "48"))


def _info(msg: str, **extras: Any) -> None:
    if extras:
        with_extras(logger, **extras).info(msg)
    else:
        logger.info(msg)


def _warn(msg: str, **extras: Any) -> None:
    if extras:
        with_extras(logger, **extras).warning(msg)
    else:
        logger.warning(msg)


def _exception(msg: str, **extras: Any) -> None:
    if extras:
        with_extras(logger, **extras).exception(msg)
    else:
        logger.exception(msg)


# -----------------------
# DOI helpers
# -----------------------

_DOI_PATTERN = re.compile(r"10\.\S+", re.IGNORECASE)


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """
    Normalize DOI to lowercase and strip URL prefixes; return None if invalid.
    """
    if not doi or not isinstance(doi, str):
        return None
    v = doi.strip().lower()
    v = re.sub(r"^https?://(dx\.)?doi\.org/", "", v)
    m = _DOI_PATTERN.search(v)
    return m.group(0) if m else None


def _extract_original_doi(payload: Any) -> Optional[str]:
    """
    Best-effort extraction of an original DOI from the FORRT response payload.
    """
    if isinstance(payload, dict):
        for key in ("original_doi", "original", "originating_doi", "origin_doi"):
            val = payload.get(key)
            norm = normalize_doi(val) if isinstance(val, str) else None
            if norm:
                return norm
        # Walk nested structures
        for val in payload.values():
            found = _extract_original_doi(val)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _extract_original_doi(item)
            if found:
                return found
    elif isinstance(payload, str):
        return normalize_doi(payload)
    return None


def _prune_nulls(obj: Any):
    """
    Recursively drop None/null entries and empty containers.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            cleaned = _prune_nulls(v)
            if cleaned is not None:
                new[k] = cleaned
        return new or None
    if isinstance(obj, list):
        cleaned_items = []
        for v in obj:
            cleaned = _prune_nulls(v)
            if cleaned is not None:
                cleaned_items.append(cleaned)
        return cleaned_items or None
    return obj


# -----------------------
# Simple file-backed cache
# -----------------------

class ForrtCache:
    def __init__(self, path: str, ttl_seconds: int):
        self.path = path
        self.ttl = ttl_seconds
        self._store: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._store = data
        except FileNotFoundError:
            self._store = {}
        except Exception as e:
            _warn("Failed to load FORRT cache", error=str(e), path=self.path)
            self._store = {}

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._store, f)
        except Exception as e:
            _warn("Failed to save FORRT cache", error=str(e), path=self.path)

    def get(self, doi: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        entry = self._store.get(doi)
        if not entry:
            return None
        ts = entry.get("ts") or 0
        if now - ts > self.ttl:
            # Expired
            return None
        return entry

    def set(self, doi: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload or {})
        payload["ts"] = time.time()
        self._store[doi] = payload


# -----------------------
# API call
# -----------------------

def _call_forrt(sess: requests.Session, doi: str, debug: bool = False) -> Dict[str, Any]:
    params = {"doi": doi}
    url = FORRT_ENDPOINT
    try:
        r = sess.get(url, params=params, timeout=25)
        if debug:
            _info("FORRT request", status=r.status_code, url=r.url)
        if r.status_code == 404:
            return {"status": "not_found", "payload": None}
        if r.status_code != 200:
            body = ""
            try:
                body = r.text[:500]
            except Exception:
                pass
            _warn("FORRT HTTP error", status=r.status_code, url=r.url, body=body)
            return {"status": f"http_{r.status_code}", "payload": None}
        try:
            js = r.json()
        except ValueError:
            _warn("FORRT invalid JSON", url=r.url)
            return {"status": "bad_json", "payload": None}
        return {"status": "ok", "payload": js}
    except requests.RequestException as e:
        _warn("FORRT network error", error=str(e), doi=doi)
        return {"status": "network_error", "payload": None}


# -----------------------
# Public entry
# -----------------------

def lookup_originals_with_forrt(
    *,
    limit: int = 200,
    osf_id: Optional[str] = None,
    ref_id: Optional[str] = None,
    only_unchecked: bool = True,
    cache_path: Optional[str] = None,
    cache_ttl_hours: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, int]:
    """
    Fetch references with DOIs from Dynamo, call FORRT original-lookup, and persist results.
    """
    repo = PreprintsRepo()
    rows = repo.select_refs_with_doi(limit=limit, osf_id=osf_id, ref_id=ref_id, only_unchecked=only_unchecked)

    ttl_hours = cache_ttl_hours if cache_ttl_hours is not None else FORRT_CACHE_TTL_HOURS_DEFAULT
    cache = ForrtCache(cache_path or FORRT_CACHE_PATH_DEFAULT, ttl_seconds=int(ttl_hours * 3600))

    stats = {"checked": 0, "updated": 0, "failed": 0, "cache_hits": 0}
    sess = requests.Session()

    for r in rows:
        osfid = r.get("osf_id")
        refid = r.get("ref_id")
        doi_raw = r.get("doi")
        doi = normalize_doi(doi_raw)
        if not doi:
            continue

        stats["checked"] += 1

        cached = cache.get(doi)
        if cached:
            stats["cache_hits"] += 1
            payload = cached.get("payload")
            payload_clean = _prune_nulls(payload)
            has_output = payload_clean is not None
            status = bool(payload_clean)
            orig = _extract_original_doi(payload_clean) if payload_clean else None
            try:
                repo.update_reference_forrt(osfid, refid, status=status, payload=payload_clean, original_doi=orig, has_output=has_output)
            except Exception:
                stats["failed"] += 1
            # refresh cache entry with cleaned payload + corrected status
            cache.set(doi, {"status": status, "payload": payload_clean, "original_doi": orig, "has_output": has_output})
            continue

        result = _call_forrt(sess, doi, debug=debug)
        payload = result.get("payload")
        status = bool(result.get("payload"))
        payload_clean = _prune_nulls(payload)
        original_doi = _extract_original_doi(payload_clean) if payload_clean else None
        has_output = payload_clean is not None

        try:
            repo.update_reference_forrt(osfid, refid, status=status, payload=payload_clean, original_doi=original_doi, has_output=has_output)
            stats["updated"] += 1
        except Exception:
            stats["failed"] += 1

        cache.set(doi, {"status": status, "payload": payload_clean, "original_doi": original_doi, "has_output": has_output})
        time.sleep(0.1)  # be polite

    cache.save()
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Lookup originals via FORRT API for references that already have DOIs.")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--osf_id", default=None)
    ap.add_argument("--ref_id", default=None)
    ap.add_argument("--no-only-unchecked", action="store_true", help="Process all DOI rows even if already checked.")
    ap.add_argument("--cache-path", default=None)
    ap.add_argument("--cache-ttl-hours", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out = lookup_originals_with_forrt(
        limit=args.limit,
        osf_id=args.osf_id,
        ref_id=args.ref_id,
        only_unchecked=not args.no_only_unchecked,
        cache_path=args.cache_path,
        cache_ttl_hours=args.cache_ttl_hours,
        debug=args.debug,
    )
    print(out)
