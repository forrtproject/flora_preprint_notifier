from __future__ import annotations
import logging
import time
import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry
from thefuzz import fuzz
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..db import engine

# -------------------
# Tunables
# -------------------
CROSSREF_BASE = "https://api.crossref.org/works"
DEFAULT_THRESHOLD = 78            # a bit stricter than OpenAlex
DEFAULT_MAX_RESULTS = 40
DEFAULT_SLEEP = 0.3
DEFAULT_ROWS = 20                 # per page
DEFAULT_UA_EMAIL = "you@example.com"
CROSSREF_CACHE_PATH = os.environ.get("CROSSREF_CACHE_PATH", os.path.join("data", "crossref_cache.json"))
CROSSREF_CACHE_TTL_HOURS = int(os.environ.get("CROSSREF_CACHE_TTL_HOURS", "24"))

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logger.setLevel(logging.INFO)


# -------------------
# Simple JSON cache
# -------------------
class _JsonCache:
    def __init__(self, path: str, ttl_hours: int):
        self.path = path
        self.ttl_seconds = max(1, ttl_hours) * 3600
        self._store: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._store = json.load(f)
        except FileNotFoundError:
            self._store = {}
        except Exception:
            self._store = {}

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._store, f)
        except Exception:
            pass

    def _expired(self, entry: dict) -> bool:
        ts = entry.get("ts") or 0
        return (time.time() - ts) > self.ttl_seconds

    def get(self, key: str) -> Optional[dict]:
        entry = self._store.get(key)
        if not entry:
            return None
        if self._expired(entry):
            return None
        return entry.get("data")

    def set(self, key: str, value: dict) -> None:
        self._store[key] = {"ts": time.time(), "data": value}


_CACHE = _JsonCache(CROSSREF_CACHE_PATH, CROSSREF_CACHE_TTL_HOURS)


# -------------------
# HTTP session
# -------------------
def make_session(user_email: str) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    # Crossref requests a descriptive UA incl. email
    s.headers.update({
        "User-Agent": f"osf_sync-crossref/1.0 (mailto:{user_email})"
    })
    return s


# -------------------
# Crossref helpers
# -------------------
def parse_crossref_results(payload: Dict) -> List[Dict]:
    items = payload.get("message", {}).get("items", []) or []
    parsed: List[Dict] = []
    for it in items:
        title = (it.get("title") or [None])[0]
        journal = (it.get("container-title") or [None])[0]
        year = None
        issued = it.get("issued", {}).get("date-parts")
        if isinstance(issued, list) and issued and isinstance(issued[0], list) and issued[0]:
            # first element of first date-parts array, e.g. [[2021, 5, 20]]
            year = issued[0][0]

        authors = []
        for a in (it.get("author") or []):
            name = " ".join([p for p in [a.get("given"), a.get("family")] if p])
            if not name:
                name = a.get("name") or ""
            authors.append(name)

        parsed.append({
            "doi": it.get("DOI"),
            "title": title,
            "journal": journal,
            "year": year,
            "authors": authors,
        })
    return parsed


def crossref_search(
    session: requests.Session,
    *,
    title: str,
    year: Optional[int],
    journal: Optional[str],
    authors: List[str],
    rows: int = DEFAULT_ROWS,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[Dict]:
    """
    Query Crossref with a bibliographic string. We bias the query with
    title + first author + journal (if present) to improve precision.
    """
    parts = [title]
    if authors:
        parts.append(authors[0])
    if journal:
        parts.append(journal)
    biblio = " ".join([p for p in parts if p]).strip()

    params = {
        "query.bibliographic": biblio,
        "rows": rows,
        "select": "DOI,title,container-title,author,issued",
    }
    if year:
        # Restrict by pub-year range around the target year
        params["filter"] = f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"

    # Cache key: use params + pagination limits
    key_raw = json.dumps(
        {
            "biblio": biblio,
            "year": year,
            "journal": journal,
            "authors": authors[:3],  # limit to keep key small
            "rows": rows,
            "max_results": max_results,
        },
        sort_keys=True,
    )
    key = hashlib.sha1(key_raw.encode("utf-8")).hexdigest()
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    results: List[Dict] = []
    cursor = "*"
    while len(results) < max_results:
        p = dict(params)
        p["cursor"] = cursor
        try:
            r = session.get(CROSSREF_BASE, params=p, timeout=30)
            data = r.json()
        except Exception as e:
            logger.warning("Crossref query failed", extra={"error": str(e), "title": title[:120]})
            break

        batch = parse_crossref_results(data)
        results.extend(batch)

        cursor = data.get("message", {}).get("next-cursor")
        if not cursor:
            break

        if len(results) < max_results:
            time.sleep(DEFAULT_SLEEP)

    out = results[:max_results]
    try:
        _CACHE.set(key, out)
        _CACHE.save()
    except Exception:
        pass
    return out


# -------------------
# Fuzzy scoring
# -------------------
def score_journal(cand: Optional[str], target: Optional[str]) -> int:
    if not cand or not target:
        return 0
    return max(
        fuzz.ratio(cand, target),
        fuzz.token_set_ratio(cand, target),
    )


def score_authors(cand: List[str], target: List[str]) -> int:
    if not cand or not target:
        return 0
    scores = []
    for ta in target:
        best = max(fuzz.token_set_ratio(ta or "", ca or "") for ca in cand)
        scores.append(best)
    return int(sum(scores) / len(scores)) if scores else 0


def pick_best(
    candidates: List[Dict],
    *,
    target_journal: Optional[str],
    target_authors: List[str],
    threshold: int,
) -> Optional[Dict]:
    best, best_score = None, -1
    for c in candidates:
        js = score_journal(c.get("journal"), target_journal)
        ascore = score_authors(c.get("authors") or [], target_authors or [])
        # Crossref: put a bit more weight on authors (often precise)
        combined = int(0.65 * ascore + 0.35 * js)
        if combined > best_score:
            best, best_score = c, combined
    if best and best.get("doi") and best_score >= threshold:
        return best
    return None


# -------------------
# DB queries
# -------------------
SEL_MISSING = text("""
SELECT osf_id, ref_id, title, authors, journal, year
FROM preprint_references
WHERE doi IS NULL
ORDER BY osf_id, ref_id
LIMIT :limit
""")

# Only set DOI if it's still NULL → idempotent and plays nice with fallback steps
UPD_CROSSREF = text("""
UPDATE preprint_references
SET doi = :doi,
    doi_source = 'crossref',
    updated_at = now()
WHERE osf_id = :osf_id AND ref_id = :ref_id AND doi IS NULL
""")


# -------------------
# Orchestrator
# -------------------
def enrich_missing_with_crossref(
    *,
    limit: int = 300,
    threshold: int = DEFAULT_THRESHOLD,
    ua_email: str = DEFAULT_UA_EMAIL,
) -> Dict[str, int]:
    """
    Fill missing DOIs in preprint_references using Crossref.
    Only updates rows where doi IS NULL, leaving any existing value intact.

    Returns: {"updated": X, "failed": Y, "checked": Z}
    """
    session = make_session(ua_email)
    updated = failed = 0

    with engine.begin() as conn:
        rows = conn.execute(SEL_MISSING, {"limit": limit}).mappings().all()

    for r in rows:
        osf_id, ref_id = r["osf_id"], r["ref_id"]
        title = r.get("title")
        authors = r.get("authors") or []
        journal = r.get("journal")
        year = r.get("year")

        if not title:
            continue

        logger.info(
            "Crossref lookup start",
            extra={"osf_id": osf_id, "ref_id": ref_id, "title": (title or "")[:120]},
        )

        try:
            cands = crossref_search(
                session,
                title=title,
                year=year if isinstance(year, int) else None,
                journal=journal,
                authors=authors,
            )
            best = pick_best(cands, target_journal=journal, target_authors=authors, threshold=threshold)
            if best and best.get("doi"):
                with engine.begin() as conn:
                    res = conn.execute(
                        UPD_CROSSREF,
                        {"doi": best["doi"], "osf_id": osf_id, "ref_id": ref_id},
                    )
                    # If someone filled DOI between SELECT and UPDATE, rowcount==0 — treat as non-failure
                    if res.rowcount:
                        updated += 1
                        logger.info(
                            "DOI matched via Crossref",
                            extra={"osf_id": osf_id, "ref_id": ref_id, "doi": best["doi"]},
                        )
                    else:
                        logger.info(
                            "Skipped update (doi already set by another step)",
                            extra={"osf_id": osf_id, "ref_id": ref_id},
                        )
            else:
                failed += 1
                logger.info(
                    "No good Crossref match",
                    extra={"osf_id": osf_id, "ref_id": ref_id, "title": (title or '')[:120]},
                )

        except SQLAlchemyError as db_e:
            failed += 1
            logger.error(
                "DB error during Crossref enrichment",
                extra={"osf_id": osf_id, "ref_id": ref_id, "error": str(db_e)},
            )
        except Exception as e:
            failed += 1
            logger.error(
                "Error in Crossref enrichment",
                extra={"osf_id": osf_id, "ref_id": ref_id, "error": str(e)},
            )

        time.sleep(DEFAULT_SLEEP)

    logger.info("Crossref enrichment complete", extra={"updated": updated, "failed": failed, "checked": len(rows)})
    return {"updated": updated, "failed": failed, "checked": len(rows)}


# -------------------
# CLI entry
# -------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Enrich missing DOIs via Crossref and update DB.")
    ap.add_argument("--limit", type=int, default=300, help="Max refs to process")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="Fuzzy acceptance threshold")
    ap.add_argument("--ua-email", default=DEFAULT_UA_EMAIL, help="Contact email to include in User-Agent")
    args = ap.parse_args()

    stats = enrich_missing_with_crossref(
        limit=args.limit,
        threshold=args.threshold,
        ua_email=args.ua_email,
    )
    print(f"✅ Crossref Enrichment Done → {stats}")
