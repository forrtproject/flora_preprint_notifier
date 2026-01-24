from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from thefuzz import fuzz

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
    OPENALEX_MAILTO,
    _fetch_candidates_relaxed,
    _fetch_candidates_strict,
    _norm,
    _safe_publication_year,
)
from osf_sync.augmentation.matching_crossref import (
    _query_crossref,
    _query_crossref_biblio,
    _safe_get_issued_year,
)
from osf_sync.dynamo.preprints_repo import PreprintsRepo

MAX_RESULTS_PER_STRATEGY = 5
THRESHOLD_STRUCTURED = 0.70
THRESHOLD_FALLBACK = 0.60
TITLE_FUZZ_THRESHOLD = 0.88  # 88% similarity required unless contained
JOURNAL_FUZZ_THRESHOLD = 0.75
SOURCE_PRIORITY = {
    "crossref_raw": 0,
    "crossref_title": 0,
    "openalex_title": 1,
}

OUTPUT_FIELDS = [
    "osf_id",
    "ref_id",
    "raw_citation",
    "title",
    "journal",
    "year",
    "authors",
    "final_doi",
    "final_score",
    "final_strategy",
    "threshold_used",
    "status",
    "best_crossref_raw_doi",
    "best_crossref_raw_score",
    "best_crossref_title_doi",
    "best_crossref_title_score",
    "best_openalex_title_doi",
    "best_openalex_title_score",
    "candidate_count",
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


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip().lower()
    for pref in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(pref):
            d = d[len(pref) :]
    return d or None


def _slug(text: Optional[str]) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _last_name_tokens(authors: List[str]) -> List[str]:
    tokens: List[str] = []
    for raw in authors:
        cleaned = re.sub(r"[,.;]", " ", raw or "")
        parts = [p for p in cleaned.split() if p]
        if parts:
            tokens.append(_slug(parts[-1]))
    return tokens


def _asymmetric_subsequence_score(needle: str, haystack: str) -> float:
    """Fraction of `needle` characters that appear in order inside `haystack`."""
    n = _slug(needle)
    h = _slug(haystack)
    if not n or not h:
        return 0.0
    it = iter(h)
    matched = 0
    for ch in n:
        for hc in it:
            if ch == hc:
                matched += 1
                break
        else:
            break
    return matched / max(1, len(n))


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


def _fetch_citation_for_doi(doi: str, style: str = "apa") -> Optional[str]:
    """Resolve a DOI into a formatted citation."""
    if not doi:
        return None
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


# ------------ candidate parsing ------------


def _extract_crossref_candidates(items: List[Dict], strategy: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        doi = _normalize_doi(it.get("DOI"))
        if not doi:
            continue
        title_list = it.get("title") or []
        journal_list = it.get("container-title") or []
        authors: List[str] = []
        for a in it.get("author") or []:
            name = " ".join([p for p in [a.get("given"), a.get("family")] if p]) or (a.get("name") or "")
            if name:
                authors.append(name)
        out.append(
            {
                "doi": doi,
                "title": title_list[0] if title_list else None,
                "journal": journal_list[0] if journal_list else None,
                "year": _safe_get_issued_year(it),
                "authors": authors,
                "source": strategy,
                "relevance": it.get("score"),
            }
        )
    return out


def _extract_openalex_candidates(items: List[Dict], strategy: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        doi = _normalize_doi(it.get("doi"))
        if not doi:
            continue
        try:
            pl = (it.get("primary_location") or {}).get("source") or {}
            journal = pl.get("display_name")
        except Exception:
            journal = None
        authors: List[str] = []
        for au in it.get("authorships") or []:
            dn = (au.get("author") or {}).get("display_name") or ""
            if dn:
                authors.append(dn)
        out.append(
            {
                "doi": doi,
                "title": it.get("title"),
                "journal": journal,
                "year": _safe_publication_year(it),
                "authors": authors,
                "source": strategy,
                "relevance": it.get("relevance_score"),
            }
        )
    return out


def _merge_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for cand in cands:
        doi = cand.get("doi")
        if not doi:
            continue
        existing = merged.get(doi)
        if not existing:
            merged[doi] = dict(cand, sources=[cand["source"]])
            continue
        merged_sources = set(existing.get("sources", []))
        merged_sources.add(cand["source"])
        priority_existing = SOURCE_PRIORITY.get(existing.get("source"), 99)
        priority_new = SOURCE_PRIORITY.get(cand["source"], 99)
        for field in ("title", "journal", "year", "relevance"):
            if cand.get(field) is None:
                continue
            if existing.get(field) is None or priority_new < priority_existing:
                existing[field] = cand[field]
        if not existing.get("authors") or priority_new < priority_existing:
            existing["authors"] = cand.get("authors") or existing.get("authors") or []
        existing["sources"] = sorted(merged_sources)
        merged[doi] = existing
    return list(merged.values())


# ------------ scoring ------------


def _title_component(ref_title: str, cand_title: Optional[str]) -> float:
    if not ref_title or not cand_title:
        return 0.0
    rt = _norm(ref_title)
    ct = _norm(cand_title)
    if not rt or not ct:
        return 0.0
    if rt in ct or ct in rt:
        return 1.0
    ratio = fuzz.token_set_ratio(rt, ct) / 100.0
    if ratio >= TITLE_FUZZ_THRESHOLD:
        return ratio
    return 0.0


def _journal_component(ref_journal: Optional[str], cand_journal: Optional[str]) -> float:
    if not ref_journal or not cand_journal:
        return 0.0
    return fuzz.token_set_ratio(ref_journal.lower(), cand_journal.lower()) / 100.0


def _year_component(ref_year: Optional[int], cand_year: Optional[int]) -> float:
    if ref_year is None or cand_year is None:
        return 0.0
    diff = abs(int(ref_year) - int(cand_year))
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.7
    return 0.0


def _author_component(ref_authors: List[str], cand_authors: List[str]) -> float:
    ref_tokens = set(_last_name_tokens(ref_authors))
    cand_tokens = set(_last_name_tokens(cand_authors))
    if not ref_tokens or not cand_tokens:
        return 0.0
    overlap = len(ref_tokens & cand_tokens)
    return overlap / max(1, len(ref_tokens))


def _score_structured(ref: Dict[str, Any], cand: Dict[str, Any]) -> float:
    title_score = _title_component(ref.get("title", ""), cand.get("title"))
    if title_score == 0.0:
        return 0.0
    weights = {
        "title": 0.55,
        "year": 0.2,
        "journal": 0.15,
        "authors": 0.10,
    }
    scores: Dict[str, float] = {"title": title_score}
    journal_score = _journal_component(ref.get("journal"), cand.get("journal"))
    if journal_score > 0:
        scores["journal"] = journal_score
    year_score = _year_component(ref.get("year"), cand.get("year"))
    if year_score > 0:
        scores["year"] = year_score
    author_score = _author_component(ref.get("authors") or [], cand.get("authors") or [])
    if author_score > 0:
        scores["authors"] = author_score

    total_weight = sum(weights[k] for k in scores.keys())
    weighted = sum(weights[k] * scores[k] for k in scores.keys())
    if total_weight == 0:
        return 0.0
    return weighted / total_weight


def _score_raw_fallback(raw_citation: str, cand: Dict[str, Any]) -> float:
    citation = _cached(f"cite::{cand.get('doi')}", _fetch_citation_for_doi, cand.get("doi"))
    if not citation:
        return 0.0
    return _asymmetric_subsequence_score(raw_citation, citation)


def _score_candidate(ref: Dict[str, Any], cand: Dict[str, Any], use_structured: bool) -> float:
    if use_structured:
        return _score_structured(ref, cand)
    return _score_raw_fallback(ref.get("raw_citation", ""), cand)


def _pick_best_candidate(
    ref: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    *,
    use_structured: bool,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Pick the best candidate by score; if scores are within a small margin,
    favor lower SOURCE_PRIORITY (CrossRef) and then higher relevance.
    """
    best = None
    best_score = -1.0
    MARGIN = 0.02  # treat scores within 0.02 as ties for source-priority purposes
    for cand in candidates:
        sc = _score_candidate(ref, cand, use_structured)
        if sc > best_score + MARGIN:
            best = cand
            best_score = sc
            continue
        if abs(sc - best_score) <= MARGIN and best is not None:
            pri_new = SOURCE_PRIORITY.get(cand.get("source"), 99)
            pri_old = SOURCE_PRIORITY.get(best.get("source"), 99)
            if pri_new < pri_old:
                best = cand
                best_score = sc
                continue
            if pri_new == pri_old:
                rel_new = cand.get("relevance")
                rel_old = best.get("relevance")
                if rel_new is not None and rel_old is not None and rel_new > rel_old:
                    best = cand
                    best_score = sc
    return best, best_score


# ------------ search runners ------------


def _search_crossref_raw(raw_citation: str, debug: bool) -> List[Dict[str, Any]]:
    if not raw_citation:
        return []
    items = _cached(f"cr_raw::{raw_citation}", _query_crossref_biblio, blob=raw_citation, rows=MAX_RESULTS_PER_STRATEGY, debug=debug) or []
    return _extract_crossref_candidates(items[:MAX_RESULTS_PER_STRATEGY], "crossref_raw")


def _search_crossref_title(
    title: str,
    year: Optional[int],
    journal: Optional[str],
    authors: List[str],
    debug: bool,
) -> List[Dict[str, Any]]:
    if not title:
        return []
    items = _cached(
        f"cr_title::{title}::{year}::{journal}::{authors}",
        _query_crossref,
        title=title,
        year=year,
        journal=journal,
        authors=authors,
        rows=MAX_RESULTS_PER_STRATEGY,
        debug=debug,
    ) or []
    return _extract_crossref_candidates(items[:MAX_RESULTS_PER_STRATEGY], "crossref_title")


def _search_openalex_title(
    sess: requests.Session,
    title: str,
    year: Optional[int],
    mailto: str,
    debug: bool,
) -> List[Dict[str, Any]]:
    if not title:
        return []
    nt = _norm(title)
    try:
        cands = _fetch_candidates_strict(sess, nt, year, debug=debug)
    except Exception:
        cands = []
    if not cands:
        cands = _fetch_candidates_relaxed(sess, nt, year, mailto, keep_year=True)
    if not cands:
        cands = _fetch_candidates_relaxed(sess, nt, None, mailto, keep_year=False)
    return _extract_openalex_candidates((cands or [])[:MAX_RESULTS_PER_STRATEGY], "openalex_title")


# ------------ processing ------------


def _process_reference(ref: Dict[str, Any], debug: bool, mailto: str) -> Dict[str, Any]:
    raw_citation = (ref.get("raw_citation") or "").strip()
    title = (ref.get("title") or "").strip()
    journal = (ref.get("journal") or "").strip() or None
    authors = ref.get("authors") or []
    year = ref.get("year")
    try:
        year = int(year) if year is not None else None
    except Exception:
        year = None

    use_structured = bool(title)
    threshold = THRESHOLD_STRUCTURED if use_structured else THRESHOLD_FALLBACK

    sess = requests.Session()

    candidates: List[Dict[str, Any]] = []
    cr_raw = _search_crossref_raw(raw_citation, debug)
    cr_title = _search_crossref_title(title, year, journal, authors, debug)
    oa_title = _search_openalex_title(sess, title, year, mailto, debug)
    candidates.extend(cr_raw)
    candidates.extend(cr_title)
    candidates.extend(oa_title)

    per_strategy_best: Dict[str, Tuple[Optional[Dict[str, Any]], float]] = {}
    per_strategy_best["crossref_raw"] = _pick_best_candidate(
        {"title": title, "journal": journal, "year": year, "authors": authors, "raw_citation": raw_citation},
        cr_raw,
        use_structured=use_structured,
    )
    per_strategy_best["crossref_title"] = _pick_best_candidate(
        {"title": title, "journal": journal, "year": year, "authors": authors, "raw_citation": raw_citation},
        cr_title,
        use_structured=use_structured,
    )
    per_strategy_best["openalex_title"] = _pick_best_candidate(
        {"title": title, "journal": journal, "year": year, "authors": authors, "raw_citation": raw_citation},
        oa_title,
        use_structured=use_structured,
    )

    merged_candidates = _merge_candidates(candidates)
    best_overall, best_score = _pick_best_candidate(
        {"title": title, "journal": journal, "year": year, "authors": authors, "raw_citation": raw_citation},
        merged_candidates,
        use_structured=use_structured,
    )

    status = "matched" if best_overall and best_score >= threshold else "no_match"
    final_doi = best_overall.get("doi") if status == "matched" else None
    final_strategy = None
    if status == "matched":
        # Pick the first contributing source to help with evaluation
        final_strategy = (best_overall.get("sources") or [best_overall.get("source")])[0]

    res = {
        "osf_id": ref.get("osf_id"),
        "ref_id": ref.get("ref_id"),
        "raw_citation": raw_citation,
        "title": title,
        "journal": journal,
        "year": year,
        "authors": authors,
        "final_doi": final_doi,
        "final_score": round(best_score, 3) if best_overall else None,
        "final_strategy": final_strategy,
        "threshold_used": threshold,
        "status": status,
        "best_crossref_raw_doi": (per_strategy_best["crossref_raw"][0] or {}).get("doi"),
        "best_crossref_raw_score": round(per_strategy_best["crossref_raw"][1], 3) if per_strategy_best["crossref_raw"][0] else None,
        "best_crossref_title_doi": (per_strategy_best["crossref_title"][0] or {}).get("doi"),
        "best_crossref_title_score": round(per_strategy_best["crossref_title"][1], 3) if per_strategy_best["crossref_title"][0] else None,
        "best_openalex_title_doi": (per_strategy_best["openalex_title"][0] or {}).get("doi"),
        "best_openalex_title_score": round(per_strategy_best["openalex_title"][1], 3) if per_strategy_best["openalex_title"][0] else None,
        "candidate_count": len(merged_candidates),
    }
    res = {k: _ascii_sanitize(v) if isinstance(v, str) else v for k, v in res.items()}
    return res


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Three-way DOI lookup: CrossRef raw citation, CrossRef title, and OpenAlex title. "
            "Returns per-strategy best candidates and a single final DOI decision."
        )
    )
    ap.add_argument("--title", help="Title to search")
    ap.add_argument("--raw", help="Raw citation string")
    ap.add_argument("--year", type=int)
    ap.add_argument("--journal")
    ap.add_argument("--author", action="append", dest="authors", help="Repeatable author")
    ap.add_argument("--osf-id", help="Lookup all references for this OSF preprint")
    ap.add_argument("--ref-id", help="Optional ref_id filter when using --osf-id")
    ap.add_argument("--limit", type=int, default=400, help="Max references to fetch when using --osf-id")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="Silence logs; emit only JSON rows")
    ap.add_argument("--mailto", default=None, help="Override OPENALEX_MAILTO/OPENALEX_EMAIL for OpenAlex requests")
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
            res = _process_reference(ref, args.debug, mailto)
            print(json.dumps({k: res.get(k) for k in OUTPUT_FIELDS}, ensure_ascii=True))
        return

    if not (args.title or args.raw):
        ap.error("Provide --osf-id or at least one of --title/--raw")

    ref = {
        "osf_id": None,
        "ref_id": None,
        "raw_citation": args.raw or "",
        "title": args.title or "",
        "authors": args.authors or [],
        "journal": args.journal,
        "year": args.year,
    }
    res = _process_reference(ref, args.debug, mailto)
    print(json.dumps({k: res.get(k) for k in OUTPUT_FIELDS}, ensure_ascii=True))


if __name__ == "__main__":
    main()
