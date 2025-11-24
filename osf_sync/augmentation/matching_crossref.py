from __future__ import annotations

import os
import html
import time
import json
import logging
import re
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
RAW_MIN_FUZZ = 70
RAW_MIN_SOLR_SCORE = 30.0
STRUCTURED_MIN_SOLR_SCORE = 52.0
TITLE_MIN_RATIO = 88
SBMV_THRESHOLD_DEFAULT = 75.0  # composite score cutoff for SBMV-style scoring

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


def _first_year_from_date_parts(blob: Optional[dict]) -> Optional[int]:
    if not blob:
        return None
    try:
        parts = blob.get("date-parts") or []
        if parts and isinstance(parts[0], list) and parts[0]:
            return int(parts[0][0])
    except Exception:
        return None
    return None


def _safe_get_issued_year(item: dict) -> Optional[int]:
    """
    Prefer the journal issue's published-print year when available; fall back to Crossref's
    canonical issued/published dates.
    """
    journal_issue = item.get("journal-issue") or {}
    year = _first_year_from_date_parts(journal_issue.get("published-print"))
    if year is not None:
        return year
    year = _first_year_from_date_parts(journal_issue.get("published-online"))
    if year is not None:
        return year
    year = _first_year_from_date_parts(item.get("published-print"))
    if year is not None:
        return year
    return _first_year_from_date_parts(item.get("issued"))


def _score_candidate_sbmv(
    cand: dict,
    title: Optional[str],
    journal: Optional[str],
    year: Optional[int],
    authors: Optional[List[str]],
    volume: Optional[str] = None,
    issue: Optional[str] = None,
    page: Optional[str] = None,
) -> float:
    """
    Crossref-style SBMV-inspired scoring: weighted blend of title, authors, journal,
    year, and volume/issue/page where available.
    """
    weights = {
        "title": 0.45,
        "authors": 0.2,
        "journal": 0.15,
        "year": 0.1,
        "vip": 0.1,  # volume/issue/page bucket
    }

    scores: Dict[str, float] = {}

    # Title
    ctitles = cand.get("title") or []
    ctitle = ctitles[0] if ctitles else ""
    if ctitle and title:
        scores["title"] = fuzz.token_set_ratio(ctitle, title)

    # Authors (overlap ratio of last names)
    if authors:
        ref_tokens = set(_last_name_tokens_from_strings(authors))
        cand_tokens = set(_candidate_author_last_names(cand))
        if ref_tokens and cand_tokens:
            overlap = len(ref_tokens & cand_tokens)
            scores["authors"] = 100.0 * overlap / max(1, len(ref_tokens))

    # Journal
    cjour = (cand.get("container-title") or [""])[0]
    if journal and cjour:
        if _journal_matches(cjour, journal):
            scores["journal"] = 100.0
        else:
            scores["journal"] = float(fuzz.token_set_ratio(cjour.lower(), journal.lower()))

    # Year
    cyear = _safe_get_issued_year(cand)
    if year is not None and cyear is not None:
        diff = abs(int(year) - int(cyear))
        if diff == 0:
            scores["year"] = 100.0
        elif diff == 1:
            scores["year"] = 80.0
        elif diff == 2:
            scores["year"] = 50.0
        else:
            scores["year"] = 0.0

    # Volume/Issue/Page
    vip_score = 0.0
    vip_checks = 0
    if volume:
        cv = str(cand.get("volume") or "")
        if cv:
            vip_checks += 1
            vip_score += 100.0 if _normalize_text(cv) == _normalize_text(volume) else 0.0
    if issue:
        ci = str(cand.get("issue") or "")
        if ci:
            vip_checks += 1
            vip_score += 100.0 if _normalize_text(ci) == _normalize_text(issue) else 0.0
    if page:
        cp = str(cand.get("page") or "")
        if cp:
            vip_checks += 1
            vip_score += 100.0 if _normalize_text(cp) == _normalize_text(page) else 0.0
    if vip_checks:
        scores["vip"] = vip_score / vip_checks

    # Weighted average over available components
    total_weight = 0.0
    total_score = 0.0
    for key, val in scores.items():
        w = weights.get(key, 0.0)
        if val is None:
            continue
        total_weight += w
        total_score += w * float(val)
    if total_weight == 0:
        return 0.0
    return total_score / total_weight


def _score_candidate_structured(cand: dict,
                                title: str,
                                year: Optional[int],
                                journal: Optional[str],
                                authors: Optional[List[str]] = None,
                                volume: Optional[str] = None,
                                issue: Optional[str] = None,
                                page: Optional[str] = None) -> float:
    return _score_candidate_sbmv(cand, title, journal, year, authors, volume, issue, page)


def _score_candidate_raw(cand: dict, raw_blob: str, authors: List[str]) -> int:
    if not raw_blob:
        return 0
    parts: List[str] = []
    parts.extend(cand.get("title") or [])
    parts.extend(cand.get("short-container-title") or [])
    parts.extend(cand.get("container-title") or [])
    parts.extend([f"{a.get('given', '')} {a.get('family', '')}".strip()
                  for a in (cand.get("author") or []) if a])
    # Add reversed/different orderings to capture initial vs expanded names
    for a in cand.get("author") or []:
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        if given and family:
            parts.append(f"{family} {given}")
            parts.append(f"{given[0]}. {family}" if given else family)
    parts.extend(authors or [])
    for key in ("volume", "issue", "page"):
        val = cand.get(key)
        if isinstance(val, list):
            parts.extend([str(x) for x in val if x])
        elif val:
            parts.append(str(val))
    year = _safe_get_issued_year(cand)
    if year:
        parts.append(str(year))
    doc = " ".join(p for p in parts if p)
    if not doc:
        return 0
    return fuzz.token_set_ratio(doc, raw_blob)


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    # Decode HTML entities (e.g., &amp;) before stripping punctuation/case
    unescaped = html.unescape(value)
    return re.sub(r"[^a-z0-9]+", "", unescaped.lower())


def _journal_matches(cand_journal: str, ref_journal: str, fuzz_threshold: int = 94) -> bool:
    """
    Consider journals matching if normalized strings are equal, or if fuzzy token_set_ratio
    clears a high bar to allow minor spelling variants (e.g., Specialities vs Specialties).
    """
    if not cand_journal or not ref_journal:
        return False
    if _normalize_text(cand_journal) == _normalize_text(ref_journal):
        return True
    ratio = fuzz.token_set_ratio(cand_journal.lower(), ref_journal.lower())
    return ratio >= fuzz_threshold


def _title_matches(cand_title: Optional[str], ref_title: Optional[str]) -> bool:
    if not cand_title or not ref_title:
        return False if ref_title else True
    ratio = fuzz.token_set_ratio(cand_title, ref_title)
    return ratio >= TITLE_MIN_RATIO


def _coerce_author_string(author) -> str:
    if isinstance(author, str):
        return author
    if isinstance(author, dict):
        for key in ("family", "name", "text", "literal", "S"):
            val = author.get(key)
            if val:
                return str(val)
        given = author.get("given")
        family = author.get("family")
        if given or family:
            return f"{given or ''} {family or ''}".strip()
    return str(author or "")


def _last_name_tokens_from_strings(authors: List) -> List[str]:
    tokens: List[str] = []
    for raw in authors:
        value = _coerce_author_string(raw)
        if not value:
            continue
        cleaned = re.sub(r"[,\.;]", " ", value)
        parts = [p for p in cleaned.split() if p]
        if parts:
            tokens.append(_normalize_text(parts[-1]))
    return [t for t in tokens if t]


def _candidate_author_last_names(cand: dict) -> List[str]:
    cands = []
    for a in cand.get("author") or []:
        if not isinstance(a, dict):
            cands.append(_normalize_text(str(a)))
            continue
        fam = a.get("family")
        if fam:
            cands.append(_normalize_text(fam))
            continue
        literal = a.get("name") or a.get("literal")
        if literal:
            cleaned = re.sub(r"[,\.;]", " ", literal)
            parts = [p for p in cleaned.split() if p]
            if parts:
                cands.append(_normalize_text(parts[-1]))
    return [c for c in cands if c]


def _authors_overlap(cand: dict, ref_authors: List) -> bool:
    ref_tokens = _last_name_tokens_from_strings(ref_authors)
    cand_tokens = _candidate_author_last_names(cand)
    if not ref_tokens or not cand_tokens:
        return True
    ref_set = set(ref_tokens)
    cand_set = set(cand_tokens)
    overlap = ref_set & cand_set
    if len(ref_set) >= 3:
        return bool(overlap)
    return True


def _raw_candidate_valid(
    cand: dict,
    raw_blob: str,
    authors: List[str],
    journal: Optional[str],
    year: Optional[int],
) -> bool:
    raw_score = _score_candidate_raw(cand, raw_blob, authors)
    if raw_score < RAW_MIN_FUZZ:
        _info("Raw score too low", raw_score=raw_score, doi=cand.get("DOI"))
        return False
    cyear = _safe_get_issued_year(cand)
    if year is not None and cyear is not None and int(year) != int(cyear):
        _info("Year mismatch in raw candidate", candidate_year=cyear, ref_year=year, doi=cand.get("DOI"))
        return False
    cjour = (cand.get("container-title") or [""])[0]
    if journal and cjour:
        if not _journal_matches(cjour, journal):
            _info("Journal mismatch in raw candidate", doi=cand.get("DOI"), candidate_journal=cjour, ref_journal=journal)
            return False
    if authors:
        if not _authors_overlap(cand, authors):
            _info("Author mismatch in raw candidate", doi=cand.get("DOI"), authors=authors)
            return False
    return True


def _structured_candidate_valid(
    cand: dict,
    title: Optional[str],
    journal: Optional[str],
    year: Optional[int],
    authors: Optional[List[str]],
) -> bool:
    ctitles = cand.get("title") or []
    ctitle = ctitles[0] if ctitles else ""
    if title:
        if not _title_matches(ctitle, title):
            return False
    cjour = (cand.get("container-title") or [""])[0]
    if journal and cjour:
        if not _journal_matches(cjour, journal):
            return False
    cyear = _safe_get_issued_year(cand)
    if year is not None and cyear is not None and int(year) != int(cyear):
        return False
    if authors:
        if not _authors_overlap(cand, authors):
            return False
    return True


def _pick_best(cands: List[dict],
               title: str,
               year: Optional[int],
               journal: Optional[str],
               threshold: int,
               debug: bool = False,
               raw_blob: Optional[str] = None,
               structured_authors: Optional[List[str]] = None,
               ref_volume: Optional[str] = None,
               ref_issue: Optional[str] = None,
               ref_page: Optional[str] = None,
               raw_search: bool = False) -> Tuple[Optional[dict], Optional[str]]:
    best = None
    best_score = -1
    last_reason: Optional[str] = None

    for c in cands:
        solr_score = c.get("score")
        if raw_blob and solr_score is not None and solr_score < RAW_MIN_SOLR_SCORE:
            if debug:
                doi = c.get("DOI")
                print(f"SKIP low-solr-score DOI={doi} SOLR={solr_score:.1f}")
            last_reason = "low-solr-score"
            continue
        if raw_blob and not _raw_candidate_valid(
            c,
            raw_blob,
            structured_authors or [],
            journal,
            year,
        ):
            if debug:
                doi = c.get("DOI")
                print(f"SKIP raw-invalid DOI={doi}")
            last_reason = "raw-validation-failed"
            continue
        if not raw_blob:
            if solr_score is not None and solr_score < STRUCTURED_MIN_SOLR_SCORE:
                if debug:
                    doi = c.get("DOI")
                    print(f"SKIP low-structured-score DOI={doi} SOLR={solr_score:.1f}")
                last_reason = "low-structured-score"
                continue
            if not _structured_candidate_valid(
                c,
                title,
                journal,
                year,
                structured_authors or [],
            ):
                if debug:
                    doi = c.get("DOI")
                    print(f"SKIP structured-invalid DOI={doi}")
                last_reason = "structured-validation-failed"
                continue
        sc = solr_score
        if sc is None:
            if raw_blob:
                sc = _score_candidate_raw(c, raw_blob, structured_authors or [])
            else:
                sc = _score_candidate_structured(c, title, year, journal, structured_authors, ref_volume, ref_issue, ref_page)
        if debug:
            doi = c.get("DOI")
            cyear = _safe_get_issued_year(c)
            ctitles = c.get("title") or []
            ctitle = ctitles[0] if ctitles else ""
            if isinstance(sc, (int, float)):
                score_str = f"{sc:6.2f}"
            else:
                score_str = str(sc)
            print(f"SCORE={score_str}  YEAR={cyear!s:>4}  DOI={doi}  TITLE={ctitle}")

        if sc > best_score:
            best_score = sc
            best = c
            last_reason = None

    effective_threshold = threshold if not raw_search else max(15, threshold // 2)
    if best and best_score >= effective_threshold:
        return best, None
    return None, last_reason or "below-threshold"


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
        "query": blob,
        # "rows": str(rows),
        # "select": ",".join(["DOI", "title", "issued", "author", "container-title"]),
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
                                 debug: bool = False,
                                 dump_misses: Optional[str] = None) -> Dict[str, int]:
    """
    For references missing a DOI (or weakly populated), try Crossref.
    If osf_id/ref_id supplied, only process that reference.

    Returns: {"checked": n, "updated": u, "failed": f}
    """
    logger.info("Crossref lookup start")

    checked = updated = failed = 0
    misses: List[Dict[str, str]] = []
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
        if not title and not raw_citation:
            _info("Skipping Crossref search: missing title and raw citation", osf_id=r.get("osf_id"), ref_id=r.get("ref_id"))
            misses.append({
                "osf_id": r.get("osf_id"),
                "ref_id": r.get("ref_id"),
                "raw_citation": raw_citation,
                "reason": "missing-title-and-raw",
            })
            continue
        raw_year = r.get("year")
        _info(raw_citation)
        year = int(raw_year) if raw_year is not None and str(raw_year).isdigit() else None
        journal = (r.get("journal") or "").strip() or None
        authors = r.get("authors") or []
        volume = (r.get("volume") or "").strip() or None
        issue = (r.get("issue") or "").strip() or None
        page = (r.get("page") or "").strip() or None

        meta = {
            "osf_id": r.get("osf_id"),
            "ref_id": r.get("ref_id"),
            "title_excerpt": title[:160],
            "authors": authors[:3],
            "year": year,
            "journal": journal,
            "using_raw_only": use_raw_only,
        }
        raw_mode = use_raw_only
        if use_raw_only:
            _info("Crossref raw-only search", **meta)
            items = _query_crossref_biblio(raw_citation, rows=30, debug=debug)
        else:
            _info("Crossref structured search", **meta)
            items = _query_crossref(title, year, journal, authors, rows=30, debug=debug)
            if not items and raw_citation:
                _info("Crossref fallback to raw citation", **meta)
                items = _query_crossref_biblio(raw_citation, rows=30, debug=debug)
                raw_mode = True
        if not items:
            _info("No Crossref candidates", osf_id=r.get("osf_id"), ref_id=r.get("ref_id"))
            misses.append({
                "osf_id": r.get("osf_id"),
                "ref_id": r.get("ref_id"),
                "raw_citation": raw_citation,
                "reason": "no-candidates",
            })
            continue

        best, miss_reason = _pick_best(
            items,
            title,
            year,
            journal,
            threshold=threshold,
            debug=debug,
            raw_blob=raw_citation if raw_mode else None,
            structured_authors=authors,
            ref_volume=volume,
            ref_issue=issue,
            ref_page=page,
            raw_search=raw_mode,
        )
        if not best:
            _info("No good Crossref match", osf_id=r.get("osf_id"), ref_id=r.get("ref_id"), candidates=len(items))
            misses.append({
                "osf_id": r.get("osf_id"),
                "ref_id": r.get("ref_id"),
                "raw_citation": raw_citation,
                "reason": miss_reason or "no-good-match",
            })
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
    if dump_misses and misses:
        import csv
        try:
            with open(dump_misses, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["osf_id", "ref_id", "raw_citation", "reason"])
                writer.writeheader()
                writer.writerows(misses)
            _info("Crossref misses written", path=dump_misses, count=len(misses))
        except Exception as e:
            _warn("Failed to write Crossref misses CSV", path=dump_misses, error=str(e))
    return {"checked": checked, "updated": updated, "failed": failed}
