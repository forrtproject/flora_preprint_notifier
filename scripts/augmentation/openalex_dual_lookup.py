from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Allow running directly (python scripts/augmentation/openalex_dual_lookup.py)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure UTF-8 stdout; we still sanitize to ASCII below
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from osf_sync.augmentation.doi_check_openalex import (
    OPENALEX_BASE,
    OPENALEX_MAILTO,
    _fetch_candidates_relaxed,
    _fetch_candidates_strict,
    _norm,
    _norm_list,
    _pick_best,
)
from osf_sync.dynamo.preprints_repo import PreprintsRepo

OUTPUT_FIELDS = [
    "osf_id",
    "ref_id",
    "raw_citation",
    "title",
    "title_openalex_id",
    "title_doi",
    "title_score",
    "title_valid",
    "title_publication_year",
    "title_candidate_count",
    "raw_openalex_id",
    "raw_doi",
    "raw_score",
    "raw_valid",
    "raw_publication_year",
    "raw_candidate_count",
]

CACHE_PATH = Path(os.environ.get("OPENALEX_CACHE_PATH", "~/.cache/openalex_dual_lookup.json")).expanduser()
CACHE_TTL_SECS = float(os.environ.get("OPENALEX_CACHE_TTL_SECS", 14 * 24 * 3600))

# ------------ utilities ------------


def _ascii_sanitize(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    repl = (
        ("\u2013", "-"),
        ("\u2014", "-"),
        ("\u2012", "-"),
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u00a0", " "),
    )
    out = val
    for src, tgt in repl:
        out = out.replace(src, tgt)
    try:
        return out.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return out


class JsonCache:
    def __init__(self, path: Path, ttl_seconds: float):
        self.path = path
        self.ttl = ttl_seconds
        self.data: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        try:
            if self.path.exists():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}

    def _persist(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def get(self, key: str):
        row = self.data.get(key)
        if not row:
            return None
        if time.time() - row.get("ts", 0) > self.ttl:
            self.data.pop(key, None)
            return None
        return row.get("value")

    def set(self, key: str, value):
        self.data[key] = {"ts": time.time(), "value": value}
        self._persist()


_cache = JsonCache(CACHE_PATH, CACHE_TTL_SECS)


def _cached(key: str, fn, *args, **kwargs):
    hit = _cache.get(key)
    if hit is not None:
        return hit
    val = fn(*args, **kwargs)
    _cache.set(key, val)
    return val


def _compute_score(title: str, year: Optional[int], journal: Optional[str], authors: List[str], cand: Dict[str, Any]) -> float:
    """Mirror the weighting in doi_check_openalex._pick_best to expose the numeric score."""
    from thefuzz import fuzz

    def _last_name(s: str) -> str:
        parts = _norm(s).split()
        return parts[-1] if parts else ""

    def _author_overlap(ref_authors: List[str], cand_auths: List[str]) -> float:
        ra = {_last_name(a) for a in ref_authors if a}
        ca = {_last_name(a) for a in cand_auths if a}
        ra.discard("")
        ca.discard("")
        if not ra or not ca:
            return 0.0
        inter = len(ra & ca)
        base = max(len(ra), len(ca))
        return 100.0 * inter / base if base else 0.0

    nt = _norm(title)
    nj = _norm(journal or "")
    nauth = _norm_list(authors)

    ct = _norm(cand.get("title"))
    cy = cand.get("publication_year")
    cj = ""
    try:
        pl = cand.get("primary_location") or {}
        src = pl.get("source") or {}
        cj = _norm(src.get("display_name"))
    except Exception:
        cj = ""

    tscore = max(fuzz.ratio(nt, ct), fuzz.token_set_ratio(nt, ct))
    cauths = []
    try:
        for au in (cand.get("authorships") or []):
            dn = au.get("author", {}).get("display_name")
            if dn:
                cauths.append(dn)
    except Exception:
        pass
    ascore = _author_overlap(nauth, cauths)
    jscore = fuzz.ratio(nj, cj) if (nj and cj) else 0

    yscore = 100.0
    if year is not None and cy is not None:
        if abs(int(cy) - int(year)) > 3:
            yscore = 0.0

    total = 0.7 * tscore + 0.2 * ascore + 0.1 * jscore
    if yscore == 0.0:
        total *= 0.6
    return float(total)


def _fetch_candidates(sess: requests.Session, title: str, year: Optional[int], journal: Optional[str],
                      authors: List[str], mailto: str, debug: bool) -> List[Dict[str, Any]]:
    # Strict, then relaxed with year, then relaxed without year
    nt = _norm(title)
    nj = _norm(journal or "")
    nauth = _norm_list(authors)

    try:
        cands = _fetch_candidates_strict(sess, nt, year, nj, nauth, debug=debug)
    except Exception:
        cands = []
    if not cands:
        cands = _fetch_candidates_relaxed(sess, nt, year, mailto, keep_year=True)
    if not cands:
        cands = _fetch_candidates_relaxed(sess, nt, None, mailto, keep_year=False)
    return cands or []


def _run_search(sess: requests.Session, search_text: str, year: Optional[int], journal: Optional[str],
                authors: List[str], mailto: str, threshold: int, debug: bool, cache_tag: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    key = f"{cache_tag}::{search_text}::{year}::{journal}::{authors}"
    cands = _cached(key, _fetch_candidates, sess, search_text, year, journal, authors, mailto, debug)
    best = _pick_best(search_text, year, journal, authors, cands, threshold=threshold, year_slack=3) if cands else None
    return best, cands or []


def _process_reference(ref: Dict[str, Any], threshold: int, debug: bool, mailto: str) -> Dict[str, Any]:
    raw_citation = (ref.get("raw_citation") or "").strip()
    title = (ref.get("title") or "").strip()
    authors = ref.get("authors") or []
    journal = (ref.get("journal") or "").strip() or None
    year = ref.get("year")
    if year is not None:
        try:
            year = int(year)
        except Exception:
            year = None

    sess = requests.Session()
    # Title search (if title present; else fall back to raw)
    title_query = title or raw_citation
    best_title, title_cands = _run_search(sess, title_query, year, journal, authors, mailto, threshold, debug, cache_tag="title")
    title_score = _compute_score(title_query, year, journal, authors, best_title) if best_title else None

    # Raw citation search (only if raw text present)
    best_raw = None
    raw_cands: List[Dict[str, Any]] = []
    if raw_citation:
        best_raw, raw_cands = _run_search(sess, raw_citation, year, journal, authors, mailto, threshold, debug, cache_tag="raw")
    raw_score = _compute_score(raw_citation or title_query, year, journal, authors, best_raw) if best_raw else None

    res = {
        "osf_id": ref.get("osf_id"),
        "ref_id": ref.get("ref_id"),
        "raw_citation": raw_citation,
        "title": title_query,
        "title_openalex_id": best_title.get("id") if best_title else None,
        "title_doi": best_title.get("doi") if best_title else None,
        "title_score": title_score,
        "title_valid": bool(best_title and title_score is not None and title_score >= threshold),
        "title_publication_year": best_title.get("publication_year") if best_title else None,
        "title_candidate_count": len(title_cands),
        "raw_openalex_id": best_raw.get("id") if best_raw else None,
        "raw_doi": best_raw.get("doi") if best_raw else None,
        "raw_score": raw_score,
        "raw_valid": bool(best_raw and raw_score is not None and raw_score >= threshold),
        "raw_publication_year": best_raw.get("publication_year") if best_raw else None,
        "raw_candidate_count": len(raw_cands),
    }
    res = {k: _ascii_sanitize(v) if isinstance(v, str) else v for k, v in res.items()}
    return res


def main():
    ap = argparse.ArgumentParser(description="OpenAlex dual lookup (title/raw) producing ASCII-only JSON rows.")
    ap.add_argument("--title", help="Title to search")
    ap.add_argument("--raw", help="Raw citation string")
    ap.add_argument("--year", type=int)
    ap.add_argument("--journal")
    ap.add_argument("--author", action="append", dest="authors", help="Repeatable author")
    ap.add_argument("--osf-id", help="Lookup all references for this OSF preprint")
    ap.add_argument("--ref-id", help="Optional ref_id filter when using --osf-id")
    ap.add_argument("--limit", type=int, default=400, help="Max references to fetch when using --osf-id")
    ap.add_argument("--threshold", type=int, default=70, help="Acceptance threshold for scoring")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="Silence logs; emit only JSON rows")
    ap.add_argument("--mailto", default=None, help="Override OPENALEX_MAILTO/OPENALEX_EMAIL")
    args = ap.parse_args()

    if args.quiet:
        logging.disable(logging.CRITICAL)

    mailto = args.mailto or os.environ.get("OPENALEX_MAILTO") or os.environ.get("OPENALEX_EMAIL") or OPENALEX_MAILTO

    if args.osf_id:
        repo = PreprintsRepo()
        refs = repo.select_refs_missing_doi(
            limit=args.limit,
            osf_id=args.osf_id,
            ref_id=args.ref_id,
            include_existing=True,
        )
        for ref in refs:
            res = _process_reference(ref, args.threshold, args.debug, mailto)
            print(json.dumps(res, ensure_ascii=True))
        return

    if not (args.title or args.raw):
        ap.error("Provide --osf-id or at least one of --title/--raw")

    title = args.title or args.raw or ""
    res = _process_reference(
        {
            "osf_id": None,
            "ref_id": None,
            "raw_citation": args.raw,
            "title": title,
            "authors": args.authors or [],
            "journal": args.journal,
            "year": args.year,
        },
        args.threshold,
        args.debug,
        mailto,
    )
    print(json.dumps(res, ensure_ascii=True))


if __name__ == "__main__":
    main()
