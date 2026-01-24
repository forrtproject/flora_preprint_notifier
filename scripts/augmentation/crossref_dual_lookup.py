from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# Allow running directly (python scripts/augmentation/crossref_dual_lookup.py)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from a .env file if present
load_dotenv()

# Ensure UTF-8 stdout to avoid Windows cp1252 issues when printing JSON
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from osf_sync.augmentation.matching_crossref import (
    _pick_best,
    _query_crossref,
    _query_crossref_biblio,
    _raw_candidate_valid,
    _score_candidate_raw,
    _score_candidate_structured,
    _structured_candidate_valid,
)
from osf_sync.dynamo.preprints_repo import PreprintsRepo

CACHE_PATH = Path(os.environ.get("CROSSREF_CACHE_PATH", "~/.cache/crossref_dual_lookup.json")).expanduser()
CACHE_TTL_SECS = float(os.environ.get("CROSSREF_CACHE_TTL_SECS", 14 * 24 * 3600))
OUTPUT_FIELDS = [
    "osf_id",
    "ref_id",
    "raw_citation",
    "title",
    "raw_doi",
    "title_doi",
    "score_raw",
    "valid_raw",
    "score_title",
    "valid_title",
    "raw_doi_citation",
    "title_doi_citation",
]


class JsonCache:
    def __init__(self, path: Path, ttl_seconds: float):
        self.path = path
        self.ttl = ttl_seconds
        self.data: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}

    def _persist(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            # cache failures should not block lookups
            pass

    def get(self, key: str):
        row = self.data.get(key)
        if not row:
            return None
        if time.time() - row.get("ts", 0) > self.ttl:
            self.data.pop(key, None)
            return None
        return row.get("value")

    def set(self, key: str, value) -> None:
        self.data[key] = {"ts": time.time(), "value": value}
        self._persist()


def _cached(key: str, fn, *args, **kwargs):
    cache = _cached.cache
    hit = cache.get(key)
    if hit is not None:
        return hit
    val = fn(*args, **kwargs)
    cache.set(key, val)
    return val


_cached.cache = JsonCache(CACHE_PATH, CACHE_TTL_SECS)  # type: ignore[attr-defined]


# --- DOI citation helper ---
def _fetch_citation_for_doi(doi: str, style: str = "apa") -> Optional[str]:
    """
    Resolve a DOI to a formatted citation. Prefer the DOI resolver with
    Accept: text/x-bibliography (style=apa), fall back to Crossref transform
    endpoint if needed.
    """
    if not doi:
        return None

    # Try doi.org with Accept header (matches the provided R example)
    try:
        r = requests.get(
            f"https://doi.org/{doi}",
            headers={"Accept": f"text/x-bibliography; style={style}"},
            timeout=20,
        )
        if r.status_code == 200 and r.text:
            return r.text.strip()
    except Exception:
        pass

    # Fallback to Crossref transform endpoint
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi}/transform/text/x-bibliography",
            params={"style": style},
            timeout=20,
        )
        if r.status_code == 200 and r.text:
            return r.text.strip()
    except Exception:
        pass

    return None


# --- main lookup ---
def crossref_dual_lookup(
    *,
    raw_citation: str,
    title: str,
    year: Optional[int],
    journal: Optional[str],
    authors: List[str],
    volume: Optional[str] = None,
    issue: Optional[str] = None,
    page: Optional[str] = None,
    threshold: int = 78,
    debug: bool = False,
) -> Dict[str, Optional[str]]:
    # Title search
    title_items = _cached(
        f"title::{title}::{year}::{journal}::{authors}",
        _query_crossref,
        title=title,
        year=year,
        journal=journal,
        authors=authors,
        rows=30,
        debug=debug,
    )
    best_title, _ = _pick_best(
        title_items or [],
        title=title,
        year=year,
        journal=journal,
        threshold=threshold,
        debug=debug,
        structured_authors=authors,
        ref_volume=volume,
        ref_issue=issue,
        ref_page=page,
        raw_search=False,
    )
    title_doi = best_title.get("DOI") if best_title else None
    score_title = None
    valid_title = False
    if best_title:
        score_title = best_title.get("score") or _score_candidate_structured(
            best_title, title, year, journal, authors, volume, issue, page
        )
        valid_title = _structured_candidate_valid(best_title, title, journal, year, authors)

    # Raw citation search
    raw_items = _cached(
        f"raw::{raw_citation}",
        _query_crossref_biblio,
        blob=raw_citation,
        rows=30,
        debug=debug,
    )
    best_raw, _ = _pick_best(
        raw_items or [],
        title=title,
        year=year,
        journal=journal,
        threshold=threshold,
        debug=debug,
        raw_blob=raw_citation,
        structured_authors=authors,
        ref_volume=volume,
        ref_issue=issue,
        ref_page=page,
        raw_search=True,
    )
    raw_doi = best_raw.get("DOI") if best_raw else None
    score_raw = None
    valid_raw = False
    if best_raw:
        score_raw = best_raw.get("score") or _score_candidate_raw(best_raw, raw_citation, authors)
        valid_raw = _raw_candidate_valid(best_raw, raw_citation, authors, journal, year)

    raw_doi_citation = _cached(f"cite::{raw_doi}", _fetch_citation_for_doi, raw_doi) if raw_doi else None
    title_doi_citation = _cached(f"cite::{title_doi}", _fetch_citation_for_doi, title_doi) if title_doi else None

    return {
        "raw_citation": raw_citation,
        "title": title,
        "raw_doi": raw_doi,
        "title_doi": title_doi,
        "score_raw": score_raw,
        "valid_raw": bool(valid_raw),
        "score_title": score_title,
        "valid_title": bool(valid_title),
        "raw_doi_citation": raw_doi_citation,
        "title_doi_citation": title_doi_citation,
    }


def _normalize_year(value) -> Optional[int]:
    try:
        if value is None:
            return None
        iv = int(value)
        return iv
    except Exception:
        return None


def _process_reference(ref: Dict, threshold: int, debug: bool) -> Dict:
    raw_citation = (ref.get("raw_citation") or "").strip()
    title = (ref.get("title") or "").strip()
    authors = ref.get("authors") or []
    year = _normalize_year(ref.get("year"))
    journal = (ref.get("journal") or "").strip() or None
    volume = (ref.get("volume") or "").strip() or None
    issue = (ref.get("issue") or "").strip() or None
    page = (ref.get("page") or "").strip() or None

    result = crossref_dual_lookup(
        raw_citation=raw_citation,
        title=title,
        year=year,
        journal=journal,
        authors=authors,
        volume=volume,
        issue=issue,
        page=page,
        threshold=threshold,
        debug=debug,
    )
    result.update({
        "osf_id": ref.get("osf_id"),
        "ref_id": ref.get("ref_id"),
    })
    return {k: result.get(k) for k in OUTPUT_FIELDS}


def _ascii_sanitize(val: Optional[str]) -> Optional[str]:
    """
    Replace common UTF-8 punctuation (en/em dash, fancy quotes) with ASCII
    equivalents and drop any remaining non-ASCII bytes. This prevents mojibake
    when viewing on non-UTF-8 consoles/editors.
    """
    if val is None:
        return None
    repl = (
        ("\u2013", "-"),  # en dash
        ("\u2014", "-"),  # em dash
        ("\u2012", "-"),  # figure dash
        ("\u2018", "'"), ("\u2019", "'"),  # single quotes
        ("\u201c", '"'), ("\u201d", '"'),  # double quotes
        ("\u00a0", " "),  # nbsp
    )
    out = val
    for src, tgt in repl:
        out = out.replace(src, tgt)
    try:
        return out.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return out


def main():
    ap = argparse.ArgumentParser(description="Crossref dual lookup (title + raw) with caching.")
    ap.add_argument("--title", help="Title to search")
    ap.add_argument("--raw", help="Raw citation string")
    ap.add_argument("--year", type=int)
    ap.add_argument("--journal")
    ap.add_argument("--author", action="append", dest="authors", help="Repeatable author")
    ap.add_argument("--osf-id", help="Lookup all references for this OSF preprint")
    ap.add_argument("--ref-id", help="Optional ref_id filter when using --osf-id")
    ap.add_argument("--limit", type=int, default=400, help="Max references to fetch when using --osf-id")
    ap.add_argument("--threshold", type=int, default=78, help="Acceptance threshold for scoring")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="Silence logs; emit only JSON rows")
    args = ap.parse_args()

    if args.quiet:
        logging.disable(logging.CRITICAL)

    if args.osf_id:
        repo = PreprintsRepo()
        refs = repo.select_refs_missing_doi(
            limit=args.limit,
            osf_id=args.osf_id,
            ref_id=args.ref_id,
            include_existing=True,
        )
        for ref in refs:
            res = _process_reference(ref, args.threshold, args.debug)
            res = {k: _ascii_sanitize(v) if isinstance(v, str) else v for k, v in res.items()}
            print(json.dumps(res, ensure_ascii=True))
        return

    if not (args.title and args.raw):
        ap.error("Provide --osf-id or both --title and --raw")

    res = crossref_dual_lookup(
        raw_citation=args.raw,
        title=args.title,
        year=args.year,
        journal=args.journal,
        authors=args.authors or [],
        threshold=args.threshold,
        debug=args.debug,
    )
    res = {k: res.get(k) for k in OUTPUT_FIELDS}
    res = {k: _ascii_sanitize(v) if isinstance(v, str) else v for k, v in res.items()}
    print(json.dumps(res, ensure_ascii=True))


if __name__ == "__main__":
    main()
