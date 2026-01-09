from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import functools
import json

import requests
from dotenv import load_dotenv

# Allow running directly (python scripts/augmentation/doi_multi_method_lookup.py)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from osf_sync.augmentation.doi_check_openalex import (
    OPENALEX_MAILTO,
    _fetch_candidates_relaxed,
    _fetch_candidates_strict,
    _norm,
    _title_matches as _oa_title_matches,
    _journal_matches as _oa_journal_matches,
    _safe_publication_year as _oa_safe_publication_year,
    _authors_overlap as _oa_authors_overlap,
    _score_candidate_sbmv as _oa_score_sbmv,
    _structured_candidate_valid as _oa_structured_valid,
)
from osf_sync.augmentation.matching_crossref import (
    RAW_MIN_SOLR_SCORE,
    SBMV_THRESHOLD_DEFAULT,
    STRUCTURED_MIN_SOLR_SCORE,
    _title_matches as _cr_title_matches,
    _journal_matches as _cr_journal_matches,
    _authors_overlap as _cr_authors_overlap,
    _safe_get_issued_year as _cr_safe_get_issued_year,
    _normalize_text as _cr_normalize_text,
    _query_crossref,
    _query_crossref_biblio,
    _raw_candidate_valid,
    _score_candidate_raw,
    _score_candidate_structured,
    _structured_candidate_valid as _cr_structured_valid,
)
from osf_sync.dynamo.client import get_dynamo_resource
from osf_sync.dynamo.preprints_repo import PreprintsRepo

# Ensure UTF-8 stdout to avoid Windows cp1252 logging issues.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger.setLevel(logging.INFO)

DOI_MULTI_METHOD_CACHE_TABLE = os.environ.get("DOI_MULTI_METHOD_CACHE_TABLE", "doi_multi_method_cache")
# Default cache TTL: one week.
DOI_MULTI_METHOD_CACHE_TTL_SECS = float(os.environ.get("DOI_MULTI_METHOD_CACHE_TTL_SECS", 7 * 24 * 3600))
_CACHE_KEY_ATTR = "cache_key"
_CACHE_VALUE_ATTR = "cache_value"
_CACHE_TTL_ATTR = "ttl"
_CACHE_NONE_ATTR = "is_none"


class DynamoCache:
    """
    DynamoDB-backed cache for API responses.
    Stores JSON-serialized values with TTL.
    """

    def __init__(self, table_name: str, ttl_seconds: float):
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        self.ddb = get_dynamo_resource()
        self.table = self.ddb.Table(table_name)
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            self.table.load()
        except Exception:
            try:
                self.ddb.create_table(
                    TableName=self.table_name,
                    KeySchema=[{"AttributeName": _CACHE_KEY_ATTR, "KeyType": "HASH"}],
                    AttributeDefinitions=[{"AttributeName": _CACHE_KEY_ATTR, "AttributeType": "S"}],
                    BillingMode="PAY_PER_REQUEST",
                )
                self.table = self.ddb.Table(self.table_name)
                self.table.wait_until_exists()
            except Exception:
                # If we cannot create/load the table, cache becomes a no-op.
                pass

    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        try:
            resp = self.table.get_item(Key={_CACHE_KEY_ATTR: key}, ConsistentRead=False)
        except Exception:
            return False, None
        item = resp.get("Item")
        if not item:
            return False, None
        ttl = item.get(_CACHE_TTL_ATTR)
        if ttl is not None and float(ttl) < time.time():
            return False, None
        if item.get(_CACHE_NONE_ATTR):
            return True, None
        try:
            raw = item.get(_CACHE_VALUE_ATTR)
            if raw is None:
                return True, None
            return True, json.loads(raw)
        except Exception:
            return False, None

    def set(self, key: str, value: Optional[Any]) -> None:
        try:
            expires = int(time.time() + self.ttl_seconds)
            item = {_CACHE_KEY_ATTR: key, _CACHE_TTL_ATTR: expires}
            if value is None:
                item[_CACHE_NONE_ATTR] = True
            else:
                item[_CACHE_VALUE_ATTR] = json.dumps(value, ensure_ascii=True, default=str)
            self.table.put_item(Item=item)
        except Exception:
            pass


_dynamo_cache = DynamoCache(DOI_MULTI_METHOD_CACHE_TABLE, DOI_MULTI_METHOD_CACHE_TTL_SECS)


def _cache_key(prefix: str, payload: Any) -> str:
    """
    Build a stable cache key from a JSON-ish payload.
    """
    try:
        blob = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        blob = repr(payload)
    return f"{prefix}::{blob}"


def _cached(key: str, fn, *args, **kwargs):
    found, hit = _dynamo_cache.get(key)
    if found:
        return hit
    val = fn(*args, **kwargs)
    _dynamo_cache.set(key, val)
    return val

METHOD_PRIORITY = ["crossref_raw", "crossref_title", "openalex_title"]
CSV_FIELDS = [
    "raw_citation",
    "title",
    "raw_citation_doi",
    "raw_citation_score",
    "raw_citation_rank_source",
    "raw_citation_rank_score",
    "column_exists_raw_citation",
    "title_doi",
    "title_score",
    "title_rank_source",
    "title_rank_score",
    "openalex_doi",
    "openalex_score",
    "openalex_rank_source",
    "openalex_rank_score",
    "crossref_citation",
    "crossref_title_citation",
    "openalex_raw_citation",
    "crossref_citation_string_distance",
    "crossref_title_citation_string_distance",
    "openalex_raw_citation_string_distance",
    "final_citation_string_distance",
    "final_method",
    "final_candidate_rank",
    "final_candidate_rank_score",
    "final_score",
    "status",
    "conflict_reason",
]

def _scan_all_refs(repo: PreprintsRepo, limit: int, ref_id: Optional[str], include_existing: bool) -> List[Dict[str, Any]]:
    """
    Fetch references from Dynamo across the entire preprint_references table.
    Applies ref_id and include_existing filtering while respecting the limit.
    """
    items: List[Dict[str, Any]] = []
    last_key = None
    while len(items) < limit:
        kwargs: Dict[str, Any] = {"Limit": min(1000, limit)}
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = repo.t_refs.scan(**kwargs)
        chunk = resp.get("Items", []) or []
        for it in chunk:
            if not it:
                continue
            if ref_id and it.get("ref_id") != ref_id:
                continue
            if not include_existing:
                doi_val = (it.get("doi") or "").strip()
                if doi_val:
                    continue
            items.append(it)
            if len(items) >= limit:
                break
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
    return items[:limit]


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip().lower()
    for pref in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(pref):
            d = d[len(pref) :]
    return d or None


def _ascii_sanitize(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    try:
        return val.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return val


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _strip_urls(val: Optional[str]) -> Optional[str]:
    """Remove URLs from a string while preserving surrounding text/spacing."""
    if val is None:
        return None
    cleaned = _URL_RE.sub("", val)
    return " ".join(cleaned.split()).strip()


def _levenshtein_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance for modest-length citation strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure a is the shorter string to minimize memory
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for i, ch_b in enumerate(b, 1):
        current = [i]
        for j, ch_a in enumerate(a, 1):
            cost = 0 if ch_a == ch_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _prep_for_distance(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    cleaned = _strip_urls(_ascii_sanitize(val))
    if cleaned is None:
        return None
    return cleaned.lower()


def _string_distance(a: Optional[str], b: Optional[str]) -> Optional[int]:
    """Compute Levenshtein distance between two citations after cleaning."""
    pa = _prep_for_distance(a)
    pb = _prep_for_distance(b)
    if pa is None or pb is None:
        return None
    return _levenshtein_distance(pa, pb)


def _raw_citation_has_year(raw: str) -> bool:
    """Check if a raw citation string appears to include a 4-digit year."""
    if not raw:
        return False
    return bool(_YEAR_RE.search(raw))


def _last_name_tokens_from_strings(authors: List[Any]) -> List[str]:
    tokens: List[str] = []
    for raw in authors:
        value = str(raw or "")
        if not value:
            continue
        cleaned = re.sub(r"[,.;]", " ", value)
        parts = [p for p in cleaned.split() if p]
        if parts:
            tokens.append(parts[-1].lower())
    return [t for t in tokens if t]


def _candidate_author_names(cand: Dict[str, Any], method: str) -> List[str]:
    names: List[str] = []
    if method.startswith("crossref"):
        for au in cand.get("author") or []:
            if isinstance(au, dict):
                given = (au.get("given") or "").strip()
                family = (au.get("family") or "").strip()
                if given or family:
                    names.append(f"{given} {family}".strip())
                else:
                    literal = (au.get("literal") or "").strip()
                    if literal:
                        names.append(literal)
            else:
                val = str(au or "").strip()
                if val:
                    names.append(val)
    elif method.startswith("openalex"):
        for au in cand.get("authorships") or []:
            try:
                dn = (au.get("author") or {}).get("display_name") or ""
                if dn:
                    names.append(dn)
            except Exception:
                continue
    return names


def _author_overlap_details(ref_authors: List[Any], cand: Dict[str, Any], method: str) -> Dict[str, Any]:
    ref_tokens = _last_name_tokens_from_strings(ref_authors)
    cand_names = _candidate_author_names(cand, method)
    cand_tokens = _last_name_tokens_from_strings(cand_names)
    overlap_tokens = sorted(set(ref_tokens) & set(cand_tokens))
    if not ref_tokens:
        overlap_pct = None
    else:
        overlap = len(set(ref_tokens) & set(cand_tokens))
        overlap_pct = round(100.0 * overlap / max(1, len(set(ref_tokens))), 1)
    return {
        "ref_authors": ref_authors,
        "cand_authors": cand_names,
        "ref_last_names": ref_tokens,
        "cand_last_names": cand_tokens,
        "overlap_last_names": overlap_tokens,
        "author_overlap_pct": overlap_pct,
    }


def _candidate_core_fields(cand: Dict[str, Any], method: str) -> Dict[str, Any]:
    if method.startswith("crossref"):
        title_list = cand.get("title") or []
        ctitle = title_list[0] if title_list else ""
        cjour = (cand.get("container-title") or [""])[0]
        cyear = _cr_safe_get_issued_year(cand)
        return {
            "title": ctitle,
            "journal": cjour,
            "year": cyear,
            "volume": cand.get("volume"),
            "issue": cand.get("issue"),
            "page": cand.get("page"),
        }
    if method.startswith("openalex"):
        ctitle = cand.get("title") or ""
        cjour = ""
        try:
            pl = cand.get("primary_location") or {}
            src = pl.get("source") or {}
            cjour = src.get("display_name") or ""
        except Exception:
            cjour = ""
        cyear = _oa_safe_publication_year(cand)
        biblio = cand.get("biblio") or {}
        return {
            "title": ctitle,
            "journal": cjour,
            "year": cyear,
            "volume": biblio.get("volume"),
            "issue": biblio.get("issue"),
            "page": biblio.get("first_page") or biblio.get("page"),
        }
    return {"title": "", "journal": "", "year": None, "volume": None, "issue": None, "page": None}


def _normalize_simple(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _match_details(
    *,
    method: str,
    ref_title: str,
    ref_journal: Optional[str],
    ref_year: Optional[int],
    ref_volume: Optional[str],
    ref_issue: Optional[str],
    ref_page: Optional[str],
    ref_authors: List[Any],
    cand: Dict[str, Any],
) -> Dict[str, Any]:
    cand_fields = _candidate_core_fields(cand, method)
    cand_title = cand_fields.get("title") or ""
    cand_journal = cand_fields.get("journal") or ""
    cand_year = cand_fields.get("year")
    cand_volume = cand_fields.get("volume")
    cand_issue = cand_fields.get("issue")
    cand_page = cand_fields.get("page")

    if method.startswith("crossref"):
        title_match = _cr_title_matches(cand_title, ref_title) if ref_title else None
        journal_match = _cr_journal_matches(cand_journal, ref_journal) if ref_journal and cand_journal else None
        authors_match = _cr_authors_overlap(cand, ref_authors) if ref_authors else None
    elif method.startswith("openalex"):
        title_match = _oa_title_matches(cand_title, ref_title) if ref_title else None
        journal_match = _oa_journal_matches(cand_journal, ref_journal) if ref_journal and cand_journal else None
        authors_match = _oa_authors_overlap(cand, ref_authors) if ref_authors else None
    else:
        title_match = None
        journal_match = None
        authors_match = None

    year_match = None
    if ref_year is not None and cand_year is not None:
        year_match = int(ref_year) == int(cand_year)

    volume_match = None
    if ref_volume and cand_volume:
        volume_match = _normalize_simple(ref_volume) == _normalize_simple(cand_volume)
    issue_match = None
    if ref_issue and cand_issue:
        issue_match = _normalize_simple(ref_issue) == _normalize_simple(cand_issue)
    page_match = None
    if ref_page and cand_page:
        page_match = _normalize_simple(ref_page) == _normalize_simple(cand_page)

    return {
        "ref_title": ref_title,
        "cand_title": cand_title,
        "title_match": title_match,
        "ref_journal": ref_journal,
        "cand_journal": cand_journal,
        "journal_match": journal_match,
        "ref_year": ref_year,
        "cand_year": cand_year,
        "year_match": year_match,
        "ref_volume": ref_volume,
        "cand_volume": cand_volume,
        "volume_match": volume_match,
        "ref_issue": ref_issue,
        "cand_issue": cand_issue,
        "issue_match": issue_match,
        "ref_page": ref_page,
        "cand_page": cand_page,
        "page_match": page_match,
        "authors_match": authors_match,
    }

@functools.lru_cache(maxsize=256)
def _fetch_citation_for_doi(doi: Optional[str], style: str = "apa") -> Optional[str]:
    """Resolve a DOI into a formatted citation text."""
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


def _score_crossref_raw(
    raw_citation: str,
    authors: List[str],
    journal: Optional[str],
    year: Optional[int],
    items: List[Dict[str, Any]],
    threshold: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    scored: List[Dict[str, Any]] = []
    for idx, cand in enumerate(items, 1):
        doi = _normalize_doi(cand.get("DOI"))
        if not doi:
            continue
        solr = cand.get("score")
        if solr is not None and solr < RAW_MIN_SOLR_SCORE:
            continue
        if not _raw_candidate_valid(cand, raw_citation, authors, journal, year):
            continue
        calc_score = _score_candidate_raw(cand, raw_citation, authors)
        scored.append(
            {
                "doi": doi,
                # Only the calculated score is used for ranking/thresholding/final choice.
                "score": float(calc_score),
                # Keep the Crossref response score for CSV/auditing only.
                "query_score": float(solr) if solr is not None else None,
                "candidate": cand,
                "candidate_number": idx,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    for idx, entry in enumerate(scored, 1):
        entry["rank"] = idx  # rank within this method by score

    eff_threshold = max(15, threshold // 2)
    best = scored[0] if scored and scored[0]["score"] >= eff_threshold else None
    return best, scored


def _score_crossref_title(
    title: str,
    year: Optional[int],
    journal: Optional[str],
    authors: List[str],
    volume: Optional[str],
    issue: Optional[str],
    page: Optional[str],
    items: List[Dict[str, Any]],
    threshold: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    scored: List[Dict[str, Any]] = []
    for idx, cand in enumerate(items, 1):
        doi = _normalize_doi(cand.get("DOI"))
        if not doi:
            continue
        solr = cand.get("score")
        if solr is not None and solr < STRUCTURED_MIN_SOLR_SCORE:
            continue
        if not _cr_structured_valid(cand, title, journal, year, authors):
            continue
        calc_score = _score_candidate_structured(
            cand,
            title,
            year,
            journal,
            authors,
            volume,
            issue,
            page,
        )
        scored.append(
            {
                "doi": doi,
                # Only the calculated score is used for ranking/thresholding/final choice.
                "score": float(calc_score),
                # Keep the Crossref response score for CSV/auditing only.
                "query_score": float(solr) if solr is not None else None,
                "candidate": cand,
                "candidate_number": idx,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    for idx, entry in enumerate(scored, 1):
        entry["rank"] = idx

    best = scored[0] if scored and scored[0]["score"] >= threshold else None
    return best, scored


def _score_openalex_title(
    title: str,
    year: Optional[int],
    journal: Optional[str],
    authors: List[str],
    mailto: str,
    threshold: int,
    debug: bool,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not title:
        return None, []

    sess = requests.Session()
    nt = _norm(title)
    strict_mailto = OPENALEX_MAILTO  # _fetch_candidates_strict uses this internally
    cands = []
    strict_key = _cache_key("oa_strict", {"title": nt, "year": year, "mailto": strict_mailto})
    try:
        cands = _cached(strict_key, _fetch_candidates_strict, sess, nt, year, debug)
    except Exception:
        cands = []

    if not cands:
        keep_year_key = _cache_key("oa_relaxed_keep_year", {"title": nt, "year": year, "mailto": mailto})
        cands = _cached(keep_year_key, _fetch_candidates_relaxed, sess, nt, year, mailto, True)

    if not cands:
        no_year_key = _cache_key("oa_relaxed_no_year", {"title": nt, "year": None, "mailto": mailto})
        cands = _cached(no_year_key, _fetch_candidates_relaxed, sess, nt, None, mailto, False)

    scored: List[Dict[str, Any]] = []
    for idx, cand in enumerate(cands or [], 1):
        doi = _normalize_doi(cand.get("doi"))
        if not doi:
            continue
        if not _oa_structured_valid(cand, title, journal, year, authors):
            continue
        score = _oa_score_sbmv(cand, title, journal, year, authors, debug=debug)
        scored.append({"doi": doi, "score": float(score), "candidate": cand, "candidate_number": idx})

    scored.sort(key=lambda x: x["score"], reverse=True)
    for idx, entry in enumerate(scored, 1):
        entry["rank"] = idx

    best = scored[0] if scored and scored[0]["score"] >= threshold else None
    return best, scored


def _resolve_final_choice(candidates: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    active = [c for c in candidates if c]
    if not active:
        return None, "no_match", None

    max_score = max(c["score"] for c in active)
    top = [c for c in active if abs(c["score"] - max_score) < 1e-6]
    doi_set = {c.get("doi") for c in top if c.get("doi")}

    if len(top) > 1 and len(doi_set) > 1:
        return None, "conflict", "multiple-dois-same-score"

    if len(top) == 1:
        return top[0], "matched", None

    # Same DOI across methods with equal score: pick by method priority.
    top_sorted = sorted(top, key=lambda c: METHOD_PRIORITY.index(c["method"]))
    return top_sorted[0], "matched", None


def process_reference(
    ref: Dict[str, Any],
    *,
    threshold: int,
    mailto: str,
    debug: bool = False,
) -> Dict[str, Any]:
    raw_citation = (ref.get("raw_citation") or "").strip()
    title = (ref.get("title") or "").strip()
    journal = (ref.get("journal") or "").strip() or None
    authors = ref.get("authors") or []
    volume = (ref.get("volume") or "").strip() or None
    issue = (ref.get("issue") or "").strip() or None
    page = (ref.get("page") or "").strip() or None
    year = ref.get("year")
    try:
        year = int(year) if year is not None else None
    except Exception:
        year = None

    if raw_citation:
        raw_key = _cache_key("cr_biblio", {"blob": raw_citation.strip(), "rows": 30})
        raw_items = _cached(raw_key, _query_crossref_biblio, blob=raw_citation, rows=30, debug=debug)
    else:
        raw_items = []

    if title:
        # matching_crossref._build_params only uses the first author string, but include all for safety.
        title_key = _cache_key(
            "cr_title",
            {
                "title": title.strip(),
                "year": year,
                "journal": journal,
                "authors": [a for a in (authors or []) if a],
                "rows": 30,
            },
        )
        title_items = _cached(
            title_key,
            _query_crossref,
            title=title,
            year=year,
            journal=journal,
            authors=authors,
            rows=30,
            debug=debug,
        )
    else:
        title_items = []

    best_raw, raw_scored = _score_crossref_raw(raw_citation, authors, journal, year, raw_items or [], threshold)
    if best_raw:
        best_raw = dict(best_raw, method="crossref_raw")

    best_title, title_scored = _score_crossref_title(
        title,
        year,
        journal,
        authors,
        volume,
        issue,
        page,
        title_items or [],
        threshold,
    )
    if best_title:
        best_title = dict(best_title, method="crossref_title")

    best_oa, oa_scored = _score_openalex_title(
        title,
        year,
        journal,
        authors,
        mailto,
        threshold,
        debug,
    )
    if best_oa:
        best_oa = dict(best_oa, method="openalex_title")

    final_choice, status, conflict_reason = _resolve_final_choice([best_raw, best_title, best_oa])

    doi_raw = best_raw.get("doi") if best_raw else None
    doi_title = best_title.get("doi") if best_title else None
    doi_oa = best_oa.get("doi") if best_oa else None

    crossref_citation_raw = _cached(_cache_key("doi_cite", {"doi": doi_raw, "style": "apa"}), _fetch_citation_for_doi, doi_raw) if doi_raw else None
    crossref_title_citation_raw = _cached(_cache_key("doi_cite", {"doi": doi_title, "style": "apa"}), _fetch_citation_for_doi, doi_title) if doi_title else None
    openalex_raw_citation_raw = _cached(_cache_key("doi_cite", {"doi": doi_oa, "style": "apa"}), _fetch_citation_for_doi, doi_oa) if doi_oa else None

    crossref_citation = _ascii_sanitize(_strip_urls(crossref_citation_raw))
    crossref_title_citation = _ascii_sanitize(_strip_urls(crossref_title_citation_raw))
    openalex_raw_citation = _ascii_sanitize(_strip_urls(openalex_raw_citation_raw))

    crossref_citation_string_distance = _string_distance(raw_citation, crossref_citation)
    crossref_title_citation_string_distance = _string_distance(raw_citation, crossref_title_citation)
    openalex_raw_citation_string_distance = _string_distance(raw_citation, openalex_raw_citation)

    method_distance_map = {
        "crossref_raw": crossref_citation_string_distance,
        "crossref_title": crossref_title_citation_string_distance,
        "openalex_title": openalex_raw_citation_string_distance,
    }
    final_citation_string_distance = method_distance_map.get(final_choice.get("method")) if final_choice else None

    crossref_raw_query_score = best_raw.get("query_score") if best_raw else None
    crossref_title_query_score = best_title.get("query_score") if best_title else None

    row = {
        "raw_citation": _ascii_sanitize(raw_citation),
        "title": _ascii_sanitize(title),
        "raw_citation_doi": best_raw.get("doi") if best_raw else None,
        "raw_citation_score": best_raw.get("score") if best_raw else None,
        "raw_citation_rank_source": best_raw.get("candidate_number") if best_raw else None,
        # Rank score now reflects the Crossref response score (solr score)
        "raw_citation_rank_score": crossref_raw_query_score,
        "column_exists_raw_citation": _raw_citation_has_year(raw_citation),
        "title_doi": best_title.get("doi") if best_title else None,
        "title_score": best_title.get("score") if best_title else None,
        "title_rank_source": best_title.get("candidate_number") if best_title else None,
        "title_rank_score": crossref_title_query_score,
        "openalex_doi": best_oa.get("doi") if best_oa else None,
        "openalex_score": best_oa.get("score") if best_oa else None,
        "openalex_rank_source": best_oa.get("candidate_number") if best_oa else None,
        # OpenAlex relevance score from the API response
        "openalex_rank_score": (best_oa.get("candidate") or {}).get("relevance_score") if best_oa else None,
        "crossref_citation": crossref_citation,
        "crossref_title_citation": crossref_title_citation,
        "openalex_raw_citation": openalex_raw_citation,
        "crossref_citation_string_distance": crossref_citation_string_distance,
        "crossref_title_citation_string_distance": crossref_title_citation_string_distance,
        "openalex_raw_citation_string_distance": openalex_raw_citation_string_distance,
        "final_citation_string_distance": final_citation_string_distance,
        "final_method": final_choice.get("method") if final_choice else None,
        # candidate_number reflects the original position returned by the source API; useful for auditing
        "final_candidate_rank": final_choice.get("candidate_number") if final_choice else None,
        # rank is the score-based ordering within the chosen method
        "final_candidate_rank_score": final_choice.get("rank") if final_choice else None,
        "final_score": final_choice.get("score") if final_choice else None,
        "status": status,
        "conflict_reason": conflict_reason,
    }

    if debug:
        if final_choice:
            cand = final_choice.get("candidate") or {}
            author_detail = _author_overlap_details(authors, cand, final_choice.get("method") or "")
            match_detail = _match_details(
                method=final_choice.get("method") or "",
                ref_title=title,
                ref_journal=journal,
                ref_year=year,
                ref_volume=volume,
                ref_issue=issue,
                ref_page=page,
                ref_authors=authors,
                cand=cand,
            )
        else:
            author_detail = {
                "ref_authors": authors,
                "cand_authors": [],
                "ref_last_names": _last_name_tokens_from_strings(authors),
                "cand_last_names": [],
                "overlap_last_names": [],
                "author_overlap_pct": None,
            }
            match_detail = {
                "ref_title": title,
                "cand_title": None,
                "title_match": None,
                "ref_journal": journal,
                "cand_journal": None,
                "journal_match": None,
                "ref_year": year,
                "cand_year": None,
                "year_match": None,
                "ref_volume": volume,
                "cand_volume": None,
                "volume_match": None,
                "ref_issue": issue,
                "cand_issue": None,
                "issue_match": None,
                "ref_page": page,
                "cand_page": None,
                "page_match": None,
                "authors_match": None,
            }
        logger.info(
            "Debug scored candidates",
            extra={
                "osf_id": ref.get("osf_id"),
                "ref_id": ref.get("ref_id"),
                "raw_scored": [(c["doi"], c["score"]) for c in raw_scored[:5]],
                "title_scored": [(c["doi"], c["score"]) for c in title_scored[:5]],
                "oa_scored": [(c["doi"], c["score"]) for c in oa_scored[:5]],
                "final": row["final_method"],
                "final_doi": final_choice.get("doi") if final_choice else None,
                "final_score": final_choice.get("score") if final_choice else None,
                "final_rank_source": final_choice.get("candidate_number") if final_choice else None,
                "final_rank_score": final_choice.get("rank") if final_choice else None,
                "final_status": status,
                "final_conflict_reason": conflict_reason,
                "ref_title": title,
                "ref_raw_citation": raw_citation,
                "ref_journal": journal,
                "ref_year": year,
                "ref_volume": volume,
                "ref_issue": issue,
                "ref_page": page,
                "ref_authors": author_detail["ref_authors"],
                "cand_authors": author_detail["cand_authors"],
                "ref_last_names": author_detail["ref_last_names"],
                "cand_last_names": author_detail["cand_last_names"],
                "author_overlap_pct": author_detail["author_overlap_pct"],
                "title_match": match_detail["title_match"],
                "journal_match": match_detail["journal_match"],
                "year_match": match_detail["year_match"],
                "volume_match": match_detail["volume_match"],
                "issue_match": match_detail["issue_match"],
                "page_match": match_detail["page_match"],
                "authors_match": match_detail["authors_match"],
            },
        )
        # Emit a human-readable line to the terminal (extras may not render with default formatter).
        logger.info(
            "Selected DOI=%s method=%s score=%s overlap_pct=%s",
            final_choice.get("doi") if final_choice else None,
            final_choice.get("method") if final_choice else None,
            final_choice.get("score") if final_choice else None,
            author_detail["author_overlap_pct"],
        )
        logger.info(
            "Author overlap details ref_authors=%s cand_authors=%s ref_last=%s cand_last=%s overlap_last=%s",
            author_detail["ref_authors"],
            author_detail["cand_authors"],
            author_detail["ref_last_names"],
            author_detail["cand_last_names"],
            author_detail["overlap_last_names"],
        )
        logger.info(
            "Match details title(ref=%s cand=%s match=%s) journal(ref=%s cand=%s match=%s) "
            "year(ref=%s cand=%s match=%s) volume(ref=%s cand=%s match=%s) "
            "issue(ref=%s cand=%s match=%s) page(ref=%s cand=%s match=%s) authors_match=%s",
            match_detail["ref_title"],
            match_detail["cand_title"],
            match_detail["title_match"],
            match_detail["ref_journal"],
            match_detail["cand_journal"],
            match_detail["journal_match"],
            match_detail["ref_year"],
            match_detail["cand_year"],
            match_detail["year_match"],
            match_detail["ref_volume"],
            match_detail["cand_volume"],
            match_detail["volume_match"],
            match_detail["ref_issue"],
            match_detail["cand_issue"],
            match_detail["issue_match"],
            match_detail["ref_page"],
            match_detail["cand_page"],
            match_detail["page_match"],
            match_detail["authors_match"],
        )

    return row


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run DOI augmentation across Crossref raw, Crossref title, and OpenAlex title searches. "
            "Uses SBMV scoring, resolves conflicts, and emits a CSV summary."
        )
    )
    ap.add_argument("--osf-id", help="OSF preprint id to process")
    ap.add_argument("--ref-id", help="Optional ref_id filter when using --osf-id")
    ap.add_argument("--limit", type=int, default=400, help="Max references to fetch when using --osf-id")
    ap.add_argument("--output", default="doi_multi_method.csv", help="CSV output path")
    ap.add_argument("--threshold", type=int, default=int(SBMV_THRESHOLD_DEFAULT), help="SBMV score threshold")
    ap.add_argument("--title", help="Title to search (standalone mode)")
    ap.add_argument("--raw", help="Raw citation to search (standalone mode)")
    ap.add_argument(
        "--raw-stdin",
        action="store_true",
        help="Read raw citation from stdin (useful for pasting multi-line citations)",
    )
    ap.add_argument("--year", type=int, help="Year hint (standalone mode)")
    ap.add_argument("--journal", help="Journal hint (standalone mode)")
    ap.add_argument("--author", action="append", dest="authors", help="Repeatable author (standalone mode)")
    ap.add_argument("--mailto", default=None, help="Override OPENALEX_MAILTO/OPENALEX_EMAIL for OpenAlex requests")
    ap.add_argument("--from-db", action="store_true", help="Read references from Dynamo preprint_references")
    ap.add_argument("--include-existing", action="store_true", help="Include refs that already have a DOI (useful for re-checking)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    mailto = args.mailto or os.environ.get("OPENALEX_MAILTO") or os.environ.get("OPENALEX_EMAIL") or OPENALEX_MAILTO
    out_path = Path(args.output)

    rows: List[Dict[str, Any]] = []

    if args.raw_stdin:
        args.raw = sys.stdin.read().strip()

    if args.from_db:
        repo = PreprintsRepo()
        if not args.osf_id:
            refs = _scan_all_refs(repo, args.limit, args.ref_id, args.include_existing)
        else:
            refs = repo.select_refs_missing_doi(
                limit=args.limit,
                osf_id=args.osf_id,
                ref_id=args.ref_id,
                include_existing=args.include_existing,
            )
        for ref in refs:
            rows.append(process_reference(ref, threshold=args.threshold, mailto=mailto, debug=args.debug))
    elif args.osf_id:
        repo = PreprintsRepo()
        refs = repo.select_refs_missing_doi(
            limit=args.limit,
            osf_id=args.osf_id,
            ref_id=args.ref_id,
            include_existing=True,
        )
        for ref in refs:
            rows.append(process_reference(ref, threshold=args.threshold, mailto=mailto, debug=args.debug))
    else:
        if not (args.title or args.raw):
            ap.error("Provide --osf-id or at least one of --title/--raw")
        ref = {
            "osf_id": None,
            "ref_id": None,
            "raw_citation": args.raw or "",
            "title": args.title or "",
            "year": args.year,
            "journal": args.journal,
            "authors": args.authors or [],
            "volume": None,
            "issue": None,
            "page": None,
        }
        rows.append(process_reference(ref, threshold=args.threshold, mailto=mailto, debug=args.debug))

    _write_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
