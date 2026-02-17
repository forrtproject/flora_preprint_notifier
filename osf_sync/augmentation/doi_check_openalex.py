from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import html
import time
import unicodedata
import logging
import requests

from thefuzz import fuzz
from ..dynamo.preprints_repo import PreprintsRepo

# -----------------------
# Logging configuration
# -----------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logger.setLevel(logging.INFO)


def _log(level: int, msg: str, **extras: Any) -> None:
    # keep previous "extras={...}" style for easy grepping
    try:
        logger.log(level, f"{msg} | extras={extras}")
    except Exception:
        logger.log(level, msg)


# -----------------------
# OpenAlex config
# -----------------------
OPENALEX_MAILTO = os.environ.get("OPENALEX_EMAIL") or os.environ.get("OPENALEX_MAILTO") or "changeme@example.com"
OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_THRESHOLD_DEFAULT = 78  # align with the Crossref matching strictness
TITLE_MIN_RATIO = 88
YEAR_MAX_DIFF = 1

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_TAG_RE = re.compile(r"<[^>]+>")

# -----------------------
# Normalization helpers
# -----------------------
_PUNCT_FIX = {
    "\u2018": "'", "\u2019": "'",  # single curly quotes
    "\u201C": '"', "\u201D": '"',  # double curly quotes
    "\u2013": "-", "\u2014": "-",  # en/em dash
    "\u00A0": " ",                 # non-breaking space
}
_WS = re.compile(r"\s+")


def _asciify(s: str) -> str:
    if not s:
        return ""
    s2 = s.translate(str.maketrans(_PUNCT_FIX))
    s3 = unicodedata.normalize("NFKD", s2)
    s3 = "".join(ch for ch in s3 if not unicodedata.combining(ch))
    return s3


def _norm(x: Any) -> str:
    if not x:
        return ""
    s = _asciify(str(x)).strip().lower()
    return _WS.sub(" ", s)


def _norm_list(xs: Optional[List[str]]) -> List[str]:
    if not xs:
        return []
    return [_norm(x) for x in xs if x]


def _normalize_text(value: str) -> str:
    """Normalize for equality/fuzzy checks similar to Crossref scoring."""
    if value is None:
        return ""
    unescaped = html.unescape(value)
    return re.sub(r"[^a-z0-9]+", "", unescaped.lower())


def _journal_matches(cand_journal: str, ref_journal: str, fuzz_threshold: int = 94) -> bool:
    """Consider journals matching if normalized strings are equal or clear a high fuzzy bar."""
    if not cand_journal or not ref_journal:
        return False
    if _normalize_text(cand_journal) == _normalize_text(ref_journal):
        return True
    ratio = fuzz.token_set_ratio(cand_journal.lower(), ref_journal.lower())
    return ratio >= fuzz_threshold


def _title_matches(cand_title: Optional[str], ref_title: Optional[str]) -> bool:
    if not cand_title or not ref_title:
        return False if ref_title else True
    ref_clean = _clean_title_text(ref_title)
    cand_clean = _clean_title_text(cand_title)
    if not ref_clean or not cand_clean:
        return False if ref_title else True
    ref_tokens = set(_tokenize_simple(ref_clean))
    cand_tokens = set(_tokenize_simple(cand_clean))
    size_ratio = min(len(ref_tokens), len(cand_tokens)) / max(len(ref_tokens), len(cand_tokens)) if ref_tokens and cand_tokens else 0.0
    cand_subset = cand_tokens < ref_tokens if ref_tokens and cand_tokens else False
    if cand_subset and len(cand_tokens) <= 2 and size_ratio < 0.5:
        return False
    ratio = fuzz.token_set_ratio(cand_clean, ref_clean)
    return ratio >= TITLE_MIN_RATIO


def _last_name_tokens_from_strings(authors: List[Any]) -> List[str]:
    tokens: List[str] = []
    for raw in authors:
        value = str(raw or "")
        if not value:
            continue
        cleaned = re.sub(r"[,.;]", " ", value)
        parts = [p for p in cleaned.split() if p]
        if parts:
            tokens.append(_normalize_text(parts[-1]))
    return [t for t in tokens if t]


def _tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _subset_inflation_penalty_core(ref_text: str, cand_text: str, min_token_set_ratio: float) -> Tuple[float, Dict[str, Any]]:
    if not ref_text or not cand_text:
        return 0.0, {"reason": "empty"}
    try:
        token_set_ratio = fuzz.token_set_ratio(ref_text, cand_text)
        if token_set_ratio < min_token_set_ratio:
            return 0.0, {"token_set_ratio": token_set_ratio, "reason": "low_token_set"}
        token_sort_ratio = fuzz.token_sort_ratio(ref_text, cand_text)
    except Exception:
        return 0.0, {"reason": "fuzz_error"}

    ref_tokens = set(_tokenize_simple(ref_text))
    cand_tokens = set(_tokenize_simple(cand_text))
    if not ref_tokens or not cand_tokens:
        return 0.0, {"reason": "no_tokens"}

    ref_subset = ref_tokens < cand_tokens
    cand_subset = cand_tokens < ref_tokens
    size_ratio = min(len(ref_tokens), len(cand_tokens)) / max(len(ref_tokens), len(cand_tokens))
    penalty = 0.0

    if ref_subset and size_ratio < 0.80 and token_sort_ratio < 95:
        penalty += min(18.0, (0.80 - size_ratio) * 60.0)

    if cand_subset and size_ratio < 0.70 and token_sort_ratio < 92:
        penalty += min(12.0, (0.70 - size_ratio) * 50.0)

    GENERIC = {
        "proceedings",
        "conference",
        "symposium",
        "workshop",
        "meeting",
        "international",
        "annual",
        "edition",
        "volume",
        "vol",
        "series",
        "lecture",
        "notes",
        "springer",
        "acm",
        "ieee",
    }
    extra_tokens = []
    if ref_subset:
        extra_tokens = list(cand_tokens - ref_tokens)
    elif cand_subset:
        extra_tokens = list(ref_tokens - cand_tokens)

    generic_extra = [t for t in extra_tokens if t in GENERIC]
    if extra_tokens:
        generic_ratio = len(generic_extra) / max(1, len(extra_tokens))
        if len(extra_tokens) >= 3 and generic_ratio >= 0.40:
            penalty += 6.0
    else:
        generic_ratio = 0.0

    meta = {
        "token_set_ratio": token_set_ratio,
        "token_sort_ratio": token_sort_ratio,
        "ref_tokens": len(ref_tokens),
        "cand_tokens": len(cand_tokens),
        "size_ratio": round(size_ratio, 3),
        "ref_subset": ref_subset,
        "cand_subset": cand_subset,
        "extra_tokens": sorted(extra_tokens)[:12],
        "generic_extra_ratio": round(generic_ratio, 3),
    }
    return penalty, meta


def _subset_inflation_penalty(ref_text: str, cand_text: str) -> Tuple[float, Dict[str, Any]]:
    return _subset_inflation_penalty_core(ref_text, cand_text, min_token_set_ratio=99.0)


def _subset_title_hard_cap(ref_text: str, cand_text: str, max_score: float) -> Tuple[Optional[float], Dict[str, Any]]:
    if not ref_text or not cand_text:
        return None, {"reason": "empty"}
    try:
        token_set_ratio = fuzz.token_set_ratio(ref_text, cand_text)
        if token_set_ratio < 99:
            return None, {"token_set_ratio": token_set_ratio, "reason": "low_token_set"}
        token_sort_ratio = fuzz.token_sort_ratio(ref_text, cand_text)
    except Exception:
        return None, {"reason": "fuzz_error"}

    ref_tokens = set(_tokenize_simple(ref_text))
    cand_tokens = set(_tokenize_simple(cand_text))
    if not ref_tokens or not cand_tokens:
        return None, {"reason": "no_tokens"}
    ref_subset = ref_tokens < cand_tokens
    size_ratio = min(len(ref_tokens), len(cand_tokens)) / max(len(ref_tokens), len(cand_tokens))

    cap = None
    if ref_subset and size_ratio < 0.85 and token_sort_ratio < 95:
        cap = max_score
    meta = {
        "token_set_ratio": token_set_ratio,
        "token_sort_ratio": token_sort_ratio,
        "ref_tokens": len(ref_tokens),
        "cand_tokens": len(cand_tokens),
        "size_ratio": round(size_ratio, 3),
        "ref_subset": ref_subset,
        "cap": cap,
    }
    return cap, meta


def _year_within_window(ref_year: Optional[int], cand_year: Optional[int], max_diff: int) -> Tuple[bool, Optional[int]]:
    if ref_year is None or cand_year is None:
        return True, None
    try:
        diff = abs(int(ref_year) - int(cand_year))
    except Exception:
        return True, None
    return diff <= max_diff, diff


def _extract_year_from_raw(raw: str) -> Optional[int]:
    if not raw:
        return None
    for match in re.finditer(r"\((19|20)\d{2}\)", raw):
        try:
            return int(match.group(0).strip("()"))
        except Exception:
            break
    last_year = None
    for match in _YEAR_RE.finditer(raw):
        last_year = match.group(0)
    if not last_year:
        return None
    try:
        return int(last_year)
    except Exception:
        return None


def _clean_title_text(text: str) -> str:
    if not text:
        return ""
    unescaped = html.unescape(text)
    cleaned = _TAG_RE.sub(" ", unescaped)
    return " ".join(cleaned.split()).strip()


def _candidate_author_last_names(cand: Dict[str, Any]) -> List[str]:
    cands: List[str] = []
    for au in cand.get("authorships") or []:
        try:
            dn = (au.get("author") or {}).get("display_name") or ""
            if dn:
                cleaned = re.sub(r"[,.;]", " ", dn)
                parts = [p for p in cleaned.split() if p]
                if parts:
                    cands.append(_normalize_text(parts[-1]))
        except Exception:
            continue
    return [c for c in cands if c]


def _authors_overlap(cand: Dict[str, Any], ref_authors: List[Any]) -> bool:
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


# -----------------------
# OpenAlex small utilities
# -----------------------
def _sget(sess: requests.Session, url: str, params: Dict[str, str], timeout: int = 30) -> Optional[requests.Response]:
    r = sess.get(url, params=params, timeout=timeout)
    return r


# -----------------------
# Candidate fetchers
# -----------------------
def _fetch_candidates(
    sess: requests.Session,
    title: str,
    year: Optional[int],
    mailto: str,
    per_page: int = 5,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Query OpenAlex for works matching the title and optionally a year filter.
    Returns up to `per_page` results sorted by relevance.
    """
    filters: List[str] = [f"title.search:{title}"]
    if year:
        filters.append(f"from_publication_date:{year}-01-01")
        filters.append(f"to_publication_date:{year}-12-31")

    params: Dict[str, str] = {
        "filter": ",".join(filters),
        "per_page": str(per_page),
        "mailto": mailto,
        "sort": "relevance_score:desc",
    }

    label = "with_year" if year else "no_year"
    try:
        r = _sget(sess, f"{OPENALEX_BASE}/works", params)
        _log(logging.INFO, f"OpenAlex {label} request",
             url=(r.url if r else None), status=(r.status_code if r else None))
        if not r:
            return []
        r.raise_for_status()
        js = r.json() or {}
        return js.get("results", []) or []
    except requests.HTTPError as e:
        body = ""
        try:
            body = r.text[:1000]
        except Exception:
            pass
        _log(logging.WARNING, f"OpenAlex HTTPError ({label})",
             status=getattr(e.response, "status_code", None),
             url=getattr(e.response, "url", None),
             body_snippet=body)
        return []
    except requests.RequestException as e:
        _log(logging.WARNING, f"OpenAlex network error ({label})", error=str(e))
        return []
    except Exception as e:
        _log(logging.WARNING, f"OpenAlex unexpected error ({label})", error=str(e))
        return []


# -----------------------
# Scoring
# -----------------------
def _safe_publication_year(cand: Dict[str, Any]) -> Optional[int]:
    try:
        y = cand.get("publication_year")
        return int(y) if y is not None else None
    except Exception:
        return None


def _structured_candidate_valid(
    cand: Dict[str, Any],
    title: Optional[str],
    journal: Optional[str],
    year: Optional[int],
    authors: Optional[List[Any]],
) -> bool:
    ct = _clean_title_text(cand.get("title") or "")
    nt = _clean_title_text(title or "")
    if title and not _title_matches(ct, nt):
        return False
    cjour = ""
    try:
        pl = cand.get("primary_location") or {}
        src = pl.get("source") or {}
        cjour = src.get("display_name") or ""
    except Exception:
        cjour = ""
    if journal and cjour:
        if not _journal_matches(cjour, journal):
            return False
    cyear = _safe_publication_year(cand)
    year_ok, _ = _year_within_window(year, cyear, YEAR_MAX_DIFF)
    if not year_ok:
        return False
    if authors:
        if not _authors_overlap(cand, authors):
            return False
    return True


def _score_candidate_sbmv(
    cand: Dict[str, Any],
    title: Optional[str],
    journal: Optional[str],
    year: Optional[int],
    authors: Optional[List[Any]],
    debug: bool = False,
) -> float:
    """
    Crossref-inspired SBMV scoring adapted for OpenAlex fields.
    """
    weights = {
        "title": 0.5,
        "authors": 0.2,
        "journal": 0.15,
        "year": 0.15,
    }

    scores: Dict[str, float] = {}

    # Title
    ctitle = _clean_title_text(cand.get("title") or "")
    nt = _clean_title_text(title or "")
    title_ratio: Optional[float] = None
    if ctitle and nt:
        title_ratio = float(fuzz.token_set_ratio(ctitle, nt))
        scores["title"] = title_ratio

    # Authors (overlap ratio of last names)
    author_overlap: Optional[float] = None
    if authors:
        ref_tokens = set(_last_name_tokens_from_strings(authors))
        cand_tokens = set(_candidate_author_last_names(cand))
        if ref_tokens and cand_tokens:
            overlap = len(ref_tokens & cand_tokens)
            author_overlap = 100.0 * overlap / max(1, len(ref_tokens))
            scores["authors"] = author_overlap

    # Journal
    cjour = ""
    try:
        pl = cand.get("primary_location") or {}
        src = pl.get("source") or {}
        cjour = src.get("display_name") or ""
    except Exception:
        cjour = ""
    journal_ratio: Optional[float] = None
    if journal and cjour:
        if _journal_matches(cjour, journal):
            scores["journal"] = 100.0
        else:
            journal_ratio = float(fuzz.token_set_ratio(cjour.lower(), journal.lower()))
            scores["journal"] = journal_ratio

    # Year
    cyear = _safe_publication_year(cand)
    year_diff: Optional[int] = None
    if year is not None and cyear is not None:
        year_ok, year_diff = _year_within_window(year, cyear, YEAR_MAX_DIFF)
        if year_ok:
            if year_diff == 0:
                scores["year"] = 100.0
            elif year_diff == 1:
                scores["year"] = 80.0
        else:
            scores["year"] = 0.0

    if debug:
        try:
            detail = (
                f"title_ratio={title_ratio if title_ratio is not None else 'NA'} "
                f"authors_overlap={author_overlap if author_overlap is not None else 'NA'} "
                f"journal_ratio={journal_ratio if journal_ratio is not None else 'NA'} "
                f"year_diff={year_diff if year_diff is not None else 'NA'} "
                f"cand_year={cyear}"
            )
            comp = []
            for k in ("title", "authors", "journal", "year"):
                if k in scores:
                    comp.append(f"{k}:{scores[k]:.1f} (w {weights[k]*100:.0f}%)")
            print(f"  DEBUG details: {detail}")
            if comp:
                print(f"  DEBUG weighted: " + "; ".join(comp))
        except Exception:
            pass

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
    score = total_score / total_weight

    penalty, _ = _subset_inflation_penalty(nt, ctitle)
    if penalty:
        score = float(score) - penalty
    cap, _ = _subset_title_hard_cap(nt, ctitle, max_score=88.0)
    if cap is not None and score > cap:
        score = float(cap)

    return float(score)


def _pick_best(
    title: str,
    year: Optional[int],
    journal: Optional[str],
    authors: List[Any],
    candidates: List[Dict[str, Any]],
    threshold: int = OPENALEX_THRESHOLD_DEFAULT,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    best = None
    best_score = -1.0
    best_rel = -1.0

    for c in candidates:
        if not _structured_candidate_valid(c, title, journal, year, authors):
            continue

        sc = _score_candidate_sbmv(c, title, journal, year, authors, debug=debug)
        rel = c.get("relevance_score")

        if debug:
            try:
                print(
                    f"SCORE={sc:6.2f}  REL={rel!s:>6}  YEAR={c.get('publication_year')!s:>4}  "
                    f"DOI={c.get('doi')}  TITLE={c.get('title')}"
                )
            except Exception:
                pass

        if (sc > best_score) or (sc == best_score and rel is not None and rel > best_rel):
            best = c
            best_score = sc
            best_rel = rel if rel is not None else best_rel

    if best and best_score >= float(threshold):
        return best
    return None


# -----------------------
# DB helpers
# -----------------------
repo = PreprintsRepo()


# -----------------------
# Public entry point
# -----------------------
def enrich_missing_with_openalex(
    *,
    osf_id: Optional[str] = None,
    limit: int = 200,
    threshold: int = OPENALEX_THRESHOLD_DEFAULT,
    debug: bool = False,
    mailto: Optional[str] = None,
) -> Dict[str, int]:
    """
    For references without a DOI:
      - query OpenAlex (title+year → title only)
      - pick best candidate by score
      - update DB if DOI present

    Returns: {"checked": N, "updated": U, "failed": F}
    """
    checked = 0
    updated = 0
    failed = 0

    rows = repo.select_refs_missing_doi(limit=limit, osf_id=osf_id)
    active_mailto = mailto or OPENALEX_MAILTO

    sess = requests.Session()

    for r in rows:
        osfid = r.get("osf_id")
        refid = r.get("ref_id")
        title = (r.get("title") or "").strip()
        raw_citation = (r.get("raw_citation") or "").strip()
        if not title:
            _log(logging.INFO, "Skipping OpenAlex search: missing title", osf_id=osfid, ref_id=refid)
            checked += 1
            continue
        authors = r.get("authors") or []  # array in DB
        journal = r.get("journal")
        raw_year = _extract_year_from_raw(raw_citation)
        if raw_year is not None:
            year = raw_year
        else:
            year = r.get("year")
        try:
            year = int(year) if year is not None else None
        except Exception:
            year = None

        _log(logging.INFO, "OpenAlex lookup start",
             osf_id=osfid, ref_id=refid, title=title, year=year, journal=journal, authors=authors)

        # Normalize once
        nt = _norm(title)

        if debug:
            _log(logging.INFO, "OpenAlex debug search title",
                 osf_id=osfid, ref_id=refid, normalized_title=nt[:160], using_raw=False)

        # Stage 1: title + year
        try:
            cands = _fetch_candidates(sess, nt, year, active_mailto, debug=debug)
        except Exception as e:
            _log(logging.WARNING, "OpenAlex error (with_year)", osf_id=osfid, ref_id=refid, error=str(e))
            cands = []

        # Stage 2: title only (no year filter)
        if not cands:
            cands = _fetch_candidates(sess, nt, None, active_mailto)

        if not cands:
            _log(logging.INFO, "No good OpenAlex match", osf_id=osfid, ref_id=refid, candidates=0)
            checked += 1
            continue

        best = _pick_best(title, year, journal, authors, cands, threshold=threshold, debug=debug)
        if not best:
            _log(logging.INFO, "No candidate passed threshold",
                 osf_id=osfid, ref_id=refid, cand_count=len(cands))
            checked += 1
            continue

        doi = best.get("doi")
        if not doi:
            _log(logging.INFO, "Best candidate has no DOI", osf_id=osfid, ref_id=refid)
            checked += 1
            continue

        # Update DB if still empty
        try:
            ok = repo.update_reference_doi(osfid, refid, doi, source="openalex")
            if ok:
                updated += 1
                _log(logging.INFO, "DOI updated via OpenAlex", osf_id=osfid, ref_id=refid, doi=doi)
            else:
                _log(logging.INFO, "DOI already present, skipped", osf_id=osfid, ref_id=refid)
            checked += 1
        except Exception as e:
            failed += 1
            checked += 1
            _log(logging.WARNING, "Failed to update DOI", osf_id=osfid, ref_id=refid, error=str(e))

        # be polite to OpenAlex
        time.sleep(0.1)

    _log(logging.INFO, "OpenAlex enrichment complete", checked=checked, updated=updated, failed=failed)
    return {"checked": checked, "updated": updated, "failed": failed}


# Allow quick local test when exec'ed in the container:
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--osf_id", default=None)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--threshold", type=int, default=OPENALEX_THRESHOLD_DEFAULT)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mailto", default=None)
    args = ap.parse_args()

    out = enrich_missing_with_openalex(
        osf_id=args.osf_id,
        limit=args.limit,
        threshold=args.threshold,
        debug=args.debug,
        mailto=args.mailto,
    )
    print("✅ OpenAlex Enrichment Done →", out)