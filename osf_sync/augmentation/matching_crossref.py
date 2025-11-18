from __future__ import annotations

import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from thefuzz import fuzz
from ..dynamo.preprints_repo import PreprintsRepo
from ..logging_setup import get_logger, with_extras

logger = get_logger(__name__)


def _info(msg: str, **extras):
    if extras:
        with_extras(logger, **extras).info(msg)
    else:
        logger.info(msg)


def _warn(msg: str, **extras):
    if extras:
        with_extras(logger, **extras).warning(msg)
    else:
        logger.warning(msg)


def _exception(msg: str, **extras):
    if extras:
        with_extras(logger, **extras).exception(msg)
    else:
        logger.exception(msg)

CROSSREF_BASE = "https://api.crossref.org/works"

# -----------------------
# Utilities
# -----------------------

def _mailto() -> str:
    # Prefer dedicated CROSSREF_MAILTO; fallback to OPENALEX_EMAIL
    return os.environ.get("CROSSREF_MAILTO") or os.environ.get("OPENALEX_EMAIL") or "devnull@example.com"


def _year_window(year: Optional[int]) -> Optional[Tuple[str, str]]:
    """Return (from, until) YYYY-MM-DD tuple if year is an int, else None."""
    if year is None:
        return None
    try:
        y = int(year)
        return (f"{y}-01-01", f"{y}-12-31")
    except Exception:
        return None


def _build_params(title: str,
                  year: Optional[int],
                  journal: Optional[str],
                  authors: List[str],
                  rows: int = 30) -> Dict[str, str]:
    """
    Build Crossref query params. We bias toward title search and add year window.
    We keep it simple to avoid 400s from unsupported params.
    """
    params = {
        "query.title": title,
        "rows": str(rows),
        "select": ",".join(["DOI", "title", "issued", "author", "container-title"]),
        "mailto": _mailto(),
    }

    ywin = _year_window(year)
    if ywin:
        params["filter"] = f"from-pub-date:{ywin[0]},until-pub-date:{ywin[1]}"

    # If we have a journal, add as an additional cue (Crossref supports query.container-title)
    if journal:
        params["query.container-title"] = journal

    # If we have at least one author, add one. (Multiple authors as multiple query.author are OK)
    if authors:
        # Use the first non-empty author string
        a = next((a for a in authors if a and a.strip()), None)
        if a:
            params["query.author"] = a

    return params


def _safe_get_issued_year(item: dict) -> Optional[int]:
    try:
        issued = item.get("issued") or {}
        parts = issued.get("date-parts") or []
        if parts and isinstance(parts[0], list) and parts[0]:
            return int(parts[0][0])
    except Exception:
        pass
    return None


def _score_candidate(cand: dict,
                     title: str,
                     year: Optional[int],
                     journal: Optional[str],
                     authors: List[str]) -> int:
    """
    Combine fuzzy title, journal similarity, author overlap, and year proximity.
    Return 0..100.
    """
    score = 0

    # Title
    ctitles = cand.get("title") or []
    ctitle = ctitles[0] if ctitles else ""
    score += fuzz.token_set_ratio(ctitle, title) * 0.6  # 60%

    # Journal
    cjour_list = cand.get("container-title") or []
    cjour = cjour_list[0] if cjour_list else ""
    if journal:
        score += fuzz.token_set_ratio(cjour, journal) * 0.2  # 20%

    # Year distance
    cyear = _safe_get_issued_year(cand)
    if year and cyear:
        diff = abs(int(year) - int(cyear))
        if diff == 0:
            score += 12  # exact year bonus
        elif diff == 1:
            score += 6   # near-miss year
        # else no bonus

    # Author overlap (very light weight)
    if authors:
        c_auth = cand.get("author") or []
        c_names = []
        for a in c_auth:
            n = a.get("family") or a.get("name") or ""
            if n:
                c_names.append(n)
        # take first author only for now
        want = (authors[0] or "").strip().lower()
        if want:
            best_a = max((fuzz.partial_ratio(want, x.lower()) for x in c_names), default=0)
            score += min(best_a, 100) * 0.08  # 8%

    return int(round(score))


def _pick_best(cands: List[dict],
               title: str,
               year: Optional[int],
               journal: Optional[str],
               authors: List[str],
               threshold: int,
               debug: bool = False) -> Optional[dict]:
    best = None
    best_score = -1

    for c in cands:
        sc = _score_candidate(c, title, year, journal, authors)
        if debug:
            doi = c.get("DOI")
            cyear = _safe_get_issued_year(c)
            ctitles = c.get("title") or []
            ctitle = ctitles[0] if ctitles else ""
            print(f"SCORE={sc:3d}  YEAR={cyear!s:>4}  DOI={doi}  TITLE={ctitle}")

        if sc > best_score:
            best_score = sc
            best = c

    if best and best_score >= threshold:
        return best
    return None


def _crossref_request(params: Dict[str, str],
                      max_attempts: int = 6,
                      debug: bool = False) -> Optional[dict]:
    """
    GET Crossref with retries. Log detailed errors.
    """
    url = CROSSREF_BASE
    _info("Crossref request start", params=params)
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 200:
                _info("Crossref request success", url=r.url, attempt=attempt)
                return r.json()
            else:
                body = r.text[:600]
                # Log both via logger and console to ensure visibility
                _warn("Crossref HTTP error", status=r.status_code, url=r.url, body=body)
                if debug:
                    print("[Crossref HTTP error] Attempt {}/{}".format(attempt, max_attempts))
                    print("  Status:", r.status_code)
                    print("  URL:", r.url)
                    print("  Body:", body)
        except requests.RequestException as e:
            _warn("Crossref network error", error=str(e))
            if debug:
                print("[Crossref network error] Attempt {}/{}".format(attempt, max_attempts))
                print("  Error:", repr(e))
        time.sleep(1.2 * attempt)
    _warn("Crossref request exhausted", attempts=max_attempts, params=params)
    return None


def _query_crossref(title: str,
                    year: Optional[int],
                    journal: Optional[str],
                    authors: List[str],
                    rows: int,
                    debug: bool) -> List[dict]:
    """
    Execute Crossref query; return list of items (may be empty).
    """
    params = _build_params(title, year, journal, authors, rows=rows)
    res = _crossref_request(params, debug=debug)
    if not res:
        return []
    try:
        items = (res.get("message") or {}).get("items") or []
        return items
    except Exception:
        return []


def _query_crossref_biblio(blob: str, rows: int, debug: bool) -> List[dict]:
    params = {
        "query.bibliographic": blob,
        "rows": str(rows),
        "select": ",".join(["DOI", "title", "issued", "author", "container-title"]),
        "mailto": _mailto(),
    }
    _info("Crossref bibliographic query", blob_excerpt=blob[:200], rows=rows, debug_mode=debug)
    res = _crossref_request(params, debug=debug)
    if not res:
        return []
    try:
        items = (res.get("message") or {}).get("items") or []
        return items
    except Exception:
        return []


# -----------------------
# Public entry
# -----------------------

def enrich_missing_with_crossref(limit: int = 300,
                                 threshold: int = 78,
                                 ua_email: Optional[str] = None,
                                 osf_id: Optional[str] = None,
                                 ref_id: Optional[str] = None,
                                 debug: bool = False) -> Dict[str, int]:
    """
    For references missing a DOI (or weakly populated), try Crossref.
    If osf_id/ref_id supplied, only process that reference.

    Returns: {"checked": n, "updated": u, "failed": f}
    """
    logger.info("Crossref lookup start")

    checked = updated = failed = 0
    repo = PreprintsRepo()
    rows = repo.select_refs_missing_doi(
        limit=limit,
        osf_id=osf_id,
        ref_id=ref_id,
        include_existing=bool(ref_id and osf_id),
    )

    for r in rows:
        if ref_id and r.get("ref_id") != ref_id:
            continue
        checked += 1
        title = (r.get("title") or "").strip()
        raw_citation = (r.get("raw_citation") or "").strip()
        use_raw_only = False
        if not title and raw_citation:
            use_raw_only = True
            _info("Crossref search using raw citation only", osf_id=r.get("osf_id"), ref_id=r.get("ref_id"))
        raw_year = r.get("year")
        year = int(raw_year) if raw_year is not None and str(raw_year).isdigit() else None
        journal = (r.get("journal") or "").strip() or None
        authors = r.get("authors") or []

        meta = {
            "osf_id": r.get("osf_id"),
            "ref_id": r.get("ref_id"),
            "title_excerpt": title[:160],
            "authors": authors[:3],
            "year": year,
            "journal": journal,
            "using_raw_only": use_raw_only,
        }
        if use_raw_only:
            _info("Crossref raw-only search", **meta)
            items = _query_crossref_biblio(raw_citation, rows=30, debug=debug)
        else:
            _info("Crossref structured search", **meta)
            items = _query_crossref(title, year, journal, authors, rows=30, debug=debug)
            if not items and raw_citation:
                _info("Crossref fallback to raw citation", **meta)
                items = _query_crossref_biblio(raw_citation, rows=30, debug=debug)
        if not items:
            _info("No Crossref candidates", osf_id=r.get("osf_id"), ref_id=r.get("ref_id"))
            continue

        best = _pick_best(items, title, year, journal, authors, threshold=threshold, debug=debug)
        if not best:
            _info("No good Crossref match", osf_id=r.get("osf_id"), ref_id=r.get("ref_id"), candidates=len(items))
            continue

        doi = best.get("DOI")
        if not doi:
            _info("Best Crossref candidate missing DOI", osf_id=r.get("osf_id"), ref_id=r.get("ref_id"))
            continue

        # Update DB
        try:
            ok = repo.update_reference_doi(r.get("osf_id"), r.get("ref_id"), doi, source="crossref")
            updated += 1 if ok else 0
        except Exception:
            failed += 1
            _exception("Dynamo update error in Crossref enrichment", osf_id=r.get("osf_id"), ref_id=r.get("ref_id"), doi=doi)

    logger.info("Crossref enrichment complete")
    return {"checked": checked, "updated": updated, "failed": failed}
