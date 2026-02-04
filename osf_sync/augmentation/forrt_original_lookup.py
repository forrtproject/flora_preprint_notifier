from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from ..dynamo.preprints_repo import PreprintsRepo
from ..dynamo.api_cache_repo import ApiCacheRepo
from ..logging_setup import get_logger, with_extras

logger = get_logger(__name__)

FORRT_ENDPOINT = os.environ.get("FORRT_ORIGINAL_LOOKUP_URL", "https://rep-api.forrt.org/v1/original-lookup")
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


def _cache_key_for_doi(doi: str) -> str:
    return f"forrt_original:{doi}"


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


def _collect_strings(value: Any) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        v = value.strip()
        if v:
            out.append(v)
    elif isinstance(value, list):
        for v in value:
            out.extend(_collect_strings(v))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_collect_strings(v))
    else:
        try:
            v = str(value).strip()
            if v:
                out.append(v)
        except Exception:
            pass
    return out


def _extract_ref_objects(payload: Any) -> List[Dict[str, Optional[str]]]:
    """
    Extract array of {doi_o, doi_r, apa_ref_o, apa_ref_r} objects from the payload.
    Deduplicates exact tuples.
    """
    out: List[Dict[str, Optional[str]]] = []

    def _append_pair(
        doi_o: Optional[str],
        doi_r: Optional[str],
        apa_ref_o: Optional[str],
        apa_ref_r: Optional[str],
    ) -> None:
        rec = {
            "doi_o": normalize_doi(doi_o) if doi_o else None,
            "doi_r": normalize_doi(doi_r) if doi_r else None,
            "apa_ref_o": apa_ref_o,
            "apa_ref_r": apa_ref_r,
        }
        out.append(rec)

    def _ensure_list(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        if isinstance(value, dict):
            return [value]
        return []

    def _from_record_lists(originals: Any, replications: Any) -> None:
        originals_list = _ensure_list(originals)
        replications_list = _ensure_list(replications)

        if originals_list and replications_list:
            for original in originals_list:
                for replication in replications_list:
                    _append_pair(
                        original.get("doi"),
                        replication.get("doi"),
                        original.get("apa_ref"),
                        replication.get("apa_ref"),
                    )
        else:
            for original in originals_list:
                _append_pair(
                    original.get("doi"),
                    None,
                    original.get("apa_ref"),
                    None,
                )
            for replication in replications_list:
                _append_pair(
                    None,
                    replication.get("doi"),
                    None,
                    replication.get("apa_ref"),
                )

    def _walk(obj: Any):
        if isinstance(obj, dict):
            if any(k in obj for k in ("originals", "replications")):
                _from_record_lists(
                    obj.get("originals"),
                    obj.get("replications"),
                )

            has_keys = any(k in obj for k in ("doi_o", "doi_r", "apa_ref_o", "apa_ref_r"))
            if has_keys:
                _append_pair(
                    obj.get("doi_o"),
                    obj.get("doi_r"),
                    obj.get("apa_ref_o"),
                    obj.get("apa_ref_r"),
                )
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(payload)

    # deduplicate while preserving order
    seen = set()
    uniq_out = []
    for rec in out:
        key = (rec.get("doi_o"), rec.get("doi_r"), rec.get("apa_ref_o"), rec.get("apa_ref_r"))
        if key in seen:
            continue
        seen.add(key)
        uniq_out.append(rec)
    return uniq_out


# -----------------------
# API call
# -----------------------

def _call_forrt(sess: requests.Session, doi: str, debug: bool = False) -> Dict[str, Any]:
    params = {"dois": doi}
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
    ignore_cache: bool = False,
    debug: bool = False,
) -> Dict[str, int]:
    """
    Fetch references with DOIs from Dynamo, call FORRT original-lookup, and persist results.
    Uses DynamoDB payloads as a cache when the prior check is within the TTL.
    """
    repo = PreprintsRepo()
    cache_repo = ApiCacheRepo()
    rows = repo.select_refs_with_doi(limit=limit, osf_id=osf_id, ref_id=ref_id, only_unchecked=only_unchecked)

    ttl_hours = cache_ttl_hours if cache_ttl_hours is not None else FORRT_CACHE_TTL_HOURS_DEFAULT
    ttl_seconds = int(ttl_hours * 3600)
    if cache_path:
        _warn("FORRT cache_path is ignored; using database cache", cache_path=cache_path)

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

        cache_key = _cache_key_for_doi(doi)
        cached_item = cache_repo.get(cache_key) if not ignore_cache else None
        cached_payload = cached_item.get("payload") if cached_item else None
        cached_status = cached_item.get("status") if cached_item else None
        cache_ready = (
            not ignore_cache
            and cache_repo.is_fresh(cached_item, ttl_seconds=ttl_seconds)
            and (cached_payload is not None or cached_status is False)
        )
        if cache_ready:
            stats["cache_hits"] += 1
            payload_clean = _prune_nulls(cached_payload)
            status = bool(payload_clean) if cached_status is None else bool(cached_status)
            ref_pairs = _extract_ref_objects(payload_clean) if payload_clean else []
            try:
                repo.update_reference_forrt(
                    osfid,
                    refid,
                    status=status,
                    ref_pairs=ref_pairs,
                )
            except Exception:
                stats["failed"] += 1
            continue

        result = _call_forrt(sess, doi, debug=debug)
        payload = result.get("payload")
        payload_clean = _prune_nulls(payload)
        status = bool(payload_clean)
        ref_pairs = _extract_ref_objects(payload_clean) if payload_clean else []
        try:
            cache_repo.put(
                cache_key,
                payload_clean,
                source="forrt_original",
                ttl_seconds=ttl_seconds,
                status=status,
            )
            repo.update_reference_forrt(
                osfid,
                refid,
                status=status,
                ref_pairs=ref_pairs,
            )
            stats["updated"] += 1
        except Exception:
            stats["failed"] += 1

        time.sleep(0.1)  # be polite

    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Lookup originals via FORRT API for references that already have DOIs.")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--osf_id", default=None)
    ap.add_argument("--ref_id", default=None)
    ap.add_argument("--no-only-unchecked", action="store_true", help="Process all DOI rows even if already checked.")
    ap.add_argument("--cache-path", default=None, help="Deprecated; ignored (database cache is used).")
    ap.add_argument("--cache-ttl-hours", type=int, default=None)
    ap.add_argument("--ignore-cache", action="store_true", help="Bypass database cache and call FORRT again.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out = lookup_originals_with_forrt(
        limit=args.limit,
        osf_id=args.osf_id,
        ref_id=args.ref_id,
        only_unchecked=not args.no_only_unchecked,
        cache_path=args.cache_path,
        cache_ttl_hours=args.cache_ttl_hours,
        ignore_cache=args.ignore_cache,
        debug=args.debug,
    )
    print(out)
