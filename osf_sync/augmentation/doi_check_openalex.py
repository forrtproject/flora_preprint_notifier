from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import re
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


# -----------------------
# OpenAlex small utilities
# -----------------------
def _sget(sess: requests.Session, url: str, params: Dict[str, str], timeout: int = 30) -> Optional[requests.Response]:
    r = sess.get(url, params=params, timeout=timeout)
    return r


def _resolve_source_id(sess: requests.Session, journal_name: Optional[str]) -> Optional[str]:
    """Resolve a journal/display name to OpenAlex source.id (Sxxxx)."""
    if not journal_name:
        return None
    try:
        params = {"search": journal_name, "per_page": "5", "mailto": OPENALEX_MAILTO}
        r = _sget(sess, f"{OPENALEX_BASE}/sources", params)
        _log(logging.INFO, "OpenAlex sources lookup", url=(r.url if r else None))
        if not r:
            return None
        r.raise_for_status()
        data = r.json() or {}
        for src in data.get("results", []):
            # choose best by name similarity
            dn = (src.get("display_name") or "").strip()
            if not dn:
                continue
            score = max(fuzz.ratio(_norm(journal_name), _norm(dn)),
                        fuzz.token_set_ratio(_norm(journal_name), _norm(dn)))
            if score >= 85:
                return src.get("id")  # eg "https://openalex.org/Sxxxx"
        return None
    except Exception as e:
        _log(logging.WARNING, "OpenAlex sources resolve error", error=str(e))
        return None


def _resolve_author_ids(sess: requests.Session, names: List[str]) -> List[str]:
    """Resolve a few author names to OpenAlex author IDs (Axxxx). Best-effort."""
    ids: List[str] = []
    for nm in names[:3]:  # cap to 3 to avoid many calls
        try:
            params = {"search": nm, "per_page": "3", "mailto": OPENALEX_MAILTO}
            r = _sget(sess, f"{OPENALEX_BASE}/authors", params)
            if not r:
                continue
            r.raise_for_status()
            js = r.json() or {}
            best_id = None
            best_score = 0
            for au in js.get("results", []):
                disp = (au.get("display_name") or "").strip()
                sc = max(fuzz.ratio(_norm(nm), _norm(disp)),
                         fuzz.token_set_ratio(_norm(nm), _norm(disp)))
                if sc > best_score:
                    best_score = sc
                    best_id = au.get("id")
            if best_id and best_score >= 85:
                ids.append(best_id)  # full URL like https://openalex.org/Axxxx
        except Exception as e:
            _log(logging.WARNING, "OpenAlex author resolve error", name=nm, error=str(e))
    return ids


# -----------------------
# Candidate fetchers
# -----------------------
def _fetch_candidates_strict(
    sess: requests.Session,
    title: str,
    year: Optional[int],
    journal: Optional[str],
    author_names: List[str],
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Strict query using OpenAlex 'filter=' style:
      filter=title.search:<title>,from_publication_date:YYYY-01-01,to_publication_date:YYYY-12-31,
             primary_location.source.id:Sxxxx,authorships.author.id:Axxxx,...
    """
    # Start filter with title.search:
    filters: List[str] = [f"title.search:{title}"]

    if year:
        filters.append(f"from_publication_date:{year}-01-01")
        filters.append(f"to_publication_date:{year}-12-31")

    # Resolve optional constraints
    s_id = _resolve_source_id(sess, journal) if journal else None
    if s_id:
        sid_short = s_id.rsplit("/", 1)[-1]
        filters.append(f"primary_location.source.id:{sid_short}")

    auth_ids = _resolve_author_ids(sess, author_names) if author_names else []
    for aid in auth_ids:
        aid_short = aid.rsplit("/", 1)[-1]
        filters.append(f"authorships.author.id:{aid_short}")

    params: Dict[str, str] = {
        "filter": ",".join(filters),
        "per_page": "25",
        "mailto": OPENALEX_MAILTO,
    }

    try:
        r = _sget(sess, f"{OPENALEX_BASE}/works", params)
        _log(logging.INFO, "OpenAlex strict request",
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
        _log(logging.WARNING, "OpenAlex HTTPError (strict)",
             status=getattr(e.response, "status_code", None),
             url=getattr(e.response, "url", None),
             body_snippet=body)
        return []
    except requests.RequestException as e:
        _log(logging.WARNING, "OpenAlex network error (strict)", error=str(e))
        return []
    except Exception as e:
        _log(logging.WARNING, "OpenAlex unexpected error (strict)", error=str(e))
        return []


def _fetch_candidates_relaxed(
    sess: requests.Session,
    title: str,
    year: Optional[int],
    mailto: str,
    keep_year: bool,
) -> List[Dict[str, Any]]:
    # Always start with title.search:
    filters: List[str] = [f"title.search:{title}"]
    if keep_year and year:
        filters.append(f"from_publication_date:{year}-01-01")
        filters.append(f"to_publication_date:{year}-12-31")

    params = {
        "filter": ",".join(filters),
        "per_page": "50",
        "mailto": mailto,
    }

    try:
        r = _sget(sess, f"{OPENALEX_BASE}/works", params)
        _log(logging.INFO, "OpenAlex relaxed request",
             url=(r.url if r else None), status=(r.status_code if r else None), keep_year=keep_year)
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
        _log(logging.WARNING, "OpenAlex HTTPError (relaxed)",
             status=getattr(e.response, "status_code", None),
             url=getattr(e.response, "url", None),
             body_snippet=body, keep_year=keep_year)
        return []
    except requests.RequestException as e:
        _log(logging.WARNING, "OpenAlex network error (relaxed)", error=str(e), keep_year=keep_year)
        return []
    except Exception as e:
        _log(logging.WARNING, "OpenAlex unexpected error (relaxed)", error=str(e), keep_year=keep_year)
        return []


# -----------------------
# Scoring
# -----------------------
def _last_name(s: str) -> str:
    # very simple: last token; robust enough for overlap test
    parts = _norm(s).split()
    return parts[-1] if parts else ""


def _author_overlap(ref_authors: List[str], cand_auths: List[str]) -> float:
    """Return overlap (0..100) of last-name sets."""
    ra = {_last_name(a) for a in ref_authors if a}
    ca = {_last_name(a) for a in cand_auths if a}
    ra.discard("")
    ca.discard("")
    if not ra or not ca:
        return 0.0
    inter = len(ra & ca)
    base = max(len(ra), len(ca))
    return 100.0 * inter / base if base else 0.0


def _cand_fields(c: Dict[str, Any]) -> Tuple[str, Optional[int], str]:
    title = _norm(c.get("title"))
    year = c.get("publication_year")
    jname = ""
    try:
        pl = c.get("primary_location") or {}
        src = pl.get("source") or {}
        jname = _norm(src.get("display_name"))
    except Exception:
        jname = ""
    return title, year, jname


def _pick_best(
    title: str,
    year: Optional[int],
    journal: Optional[str],
    authors: List[str],
    candidates: List[Dict[str, Any]],
    threshold: int = 70,
    year_slack: int = 3,
) -> Optional[Dict[str, Any]]:
    nt = _norm(title)
    nj = _norm(journal or "")
    nauth = _norm_list(authors)

    best = None
    best_score = -1.0

    for c in candidates:
        ct, cy, cj = _cand_fields(c)

        # Title similarity
        tscore = max(fuzz.ratio(nt, ct), fuzz.token_set_ratio(nt, ct))

        # Authors overlap (use authorships list)
        cauths = []
        try:
            for au in (c.get("authorships") or []):
                dn = au.get("author", {}).get("display_name")
                if dn:
                    cauths.append(dn)
        except Exception:
            pass
        ascore = _author_overlap(nauth, cauths)

        # Journal similarity (if both present)
        jscore = fuzz.ratio(nj, cj) if (nj and cj) else 0

        # Year compatibility
        yscore = 100.0
        if year is not None and cy is not None:
            if abs(int(cy) - int(year)) > year_slack:
                yscore = 0.0  # too far, penalize hard

        # Weighted total: emphasize title
        total = 0.7 * tscore + 0.2 * ascore + 0.1 * jscore
        if yscore == 0.0:
            total *= 0.6  # harsher if year far away

        if total > best_score:
            best = c
            best_score = total

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
    threshold: int = 70,
    debug: bool = False,
    mailto: Optional[str] = None,
) -> Dict[str, int]:
    """
    For references without a DOI:
      - query OpenAlex (strict → relaxed keep-year → relaxed no-year)
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
        using_raw = False
        if not title and raw_citation:
            title = raw_citation
            using_raw = True
            _log(logging.INFO, "Using raw citation text for OpenAlex search", osf_id=osfid, ref_id=refid)
        authors = r.get("authors") or []  # array in DB
        journal = r.get("journal")
        year = r.get("year")

        _log(logging.INFO, "OpenAlex lookup start",
             osf_id=osfid, ref_id=refid, title=title, year=year, journal=journal, authors=authors)

        # Normalize once
        nt = _norm(title)
        nj = _norm(journal or "")
        nauth = _norm_list(authors)

        if debug:
            _log(logging.INFO, "OpenAlex debug search title",
                 osf_id=osfid, ref_id=refid, normalized_title=nt[:160], using_raw=using_raw)

        # Stage 1: strict
        try:
            cands = _fetch_candidates_strict(sess, nt, year, nj, nauth, debug=debug)
        except Exception as e:
            _log(logging.WARNING, "OpenAlex error (strict)", osf_id=osfid, ref_id=refid, error=str(e))
            cands = []

        # Stage 2: relaxed keep-year (title + year window)
        if not cands:
            cands = _fetch_candidates_relaxed(sess, nt, year, active_mailto, keep_year=True)

        # Stage 3: relaxed no-year (title only)
        if not cands:
            cands = _fetch_candidates_relaxed(sess, nt, None, active_mailto, keep_year=False)

        if not cands:
            _log(logging.INFO, "No good OpenAlex match", osf_id=osfid, ref_id=refid, candidates=0)
            checked += 1
            continue

        best = _pick_best(title, year, journal, authors, cands, threshold=threshold, year_slack=3)
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
    ap.add_argument("--threshold", type=int, default=70)
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
