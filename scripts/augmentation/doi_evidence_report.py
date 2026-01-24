from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from thefuzz import fuzz

# Allow running directly (python scripts/augmentation/doi_evidence_report.py)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from osf_sync.augmentation.doi_check_openalex import (
    OPENALEX_MAILTO,
    OPENALEX_THRESHOLD_DEFAULT,
    _fetch_candidates_relaxed,
    _fetch_candidates_strict,
    _norm,
    _pick_best as oa_pick_best,
    _score_candidate_sbmv,
)
from osf_sync.augmentation.matching_crossref import (
    SBMV_THRESHOLD_DEFAULT,
    _pick_best as cr_pick_best,
    _query_crossref,
    _query_crossref_biblio,
    _raw_candidate_valid,
    _score_candidate_raw,
    _score_candidate_structured,
)
from osf_sync.dynamo.preprints_repo import PreprintsRepo

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger.setLevel(logging.INFO)


# ----------------------------
# Small helpers
# ----------------------------
def _sanitize(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    try:
        return val.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return val


def _fetch_citation_for_doi(doi: str, style: str = "apa") -> Optional[str]:
    """Resolve a DOI to a formatted citation."""
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


def _comparison_score(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b:
        return None
    return fuzz.token_set_ratio(a, b)


# ----------------------------
# Crossref search + scoring
# ----------------------------
def _run_crossref(
    *,
    raw_citation: str,
    title: str,
    year: Optional[int],
    journal: Optional[str],
    authors: List[str],
    volume: Optional[str],
    issue: Optional[str],
    page: Optional[str],
    threshold: int,
    debug: bool,
) -> Dict[str, Any]:
    # Raw search
    raw_items = _query_crossref_biblio(blob=raw_citation, rows=30, debug=debug) if raw_citation else []
    best_raw, _ = cr_pick_best(
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
    api_raw_doi = best_raw.get("DOI") if best_raw else None
    score_raw = None
    valid_raw = False
    cr_citation_raw = None
    if best_raw:
        score_raw = best_raw.get("score") or _score_candidate_raw(best_raw, raw_citation, authors)
        valid_raw = _raw_candidate_valid(best_raw, raw_citation, authors, journal, year)
        cr_citation_raw = _fetch_citation_for_doi(api_raw_doi)

    # Title search
    title_items = _query_crossref(
        title=title,
        year=year,
        journal=journal,
        authors=authors,
        rows=30,
        debug=debug,
    ) if title else []
    best_title, _ = cr_pick_best(
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
    api_title_doi = best_title.get("DOI") if best_title else None
    score_title = None
    valid_title = False
    cr_citation_title = None
    if best_title:
        score_title = best_title.get("score") or _score_candidate_structured(
            best_title,
            title,
            year,
            journal,
            authors,
            volume,
            issue,
            page,
        )
        valid_title = True  # _pick_best already enforced threshold/validity
        cr_citation_title = _fetch_citation_for_doi(api_title_doi)

    return {
        "api_raw_doi": api_raw_doi,
        "score_raw": score_raw,
        "valid_raw": bool(valid_raw),
        "api_title_doi": api_title_doi,
        "score_title": score_title,
        "valid_title": bool(valid_title),
        "cr_citation_raw": cr_citation_raw,
        "cr_citation_title": cr_citation_title,
    }


# ----------------------------
# OpenAlex search + scoring
# ----------------------------
def _run_openalex(
    *,
    title: str,
    year: Optional[int],
    journal: Optional[str],
    authors: List[str],
    mailto: str,
    debug: bool,
) -> Dict[str, Any]:
    oa_api_title_doi = None
    oa_relevancy_score = None
    oa_relevancy_score_code = None

    if not title:
        return {
            "oa_api_title_doi": oa_api_title_doi,
            "oa_relevancy_score": oa_relevancy_score,
            "oa_relevancy_score_code": oa_relevancy_score_code,
        }

    sess = requests.Session()
    nt = _norm(title)
    try:
        cands = _fetch_candidates_strict(sess, nt, year, debug=debug)
    except Exception:
        cands = []
    if not cands:
        cands = _fetch_candidates_relaxed(sess, nt, year, mailto, keep_year=True)
    if not cands:
        cands = _fetch_candidates_relaxed(sess, nt, None, mailto, keep_year=False)

    best = oa_pick_best(title, year, journal, authors, cands or [], threshold=OPENALEX_THRESHOLD_DEFAULT, debug=debug) if cands else None
    if best:
        oa_api_title_doi = best.get("doi")
        oa_relevancy_score = best.get("relevance_score")
        oa_relevancy_score_code = _score_candidate_sbmv(best, title, journal, year, authors)

    return {
        "oa_api_title_doi": oa_api_title_doi,
        "oa_relevancy_score": oa_relevancy_score,
        "oa_relevancy_score_code": oa_relevancy_score_code,
    }


# ----------------------------
# Main reporting
# ----------------------------
def process_reference(ref: Dict[str, Any], mailto: str, threshold: int, debug: bool) -> Dict[str, Any]:
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

    cr = _run_crossref(
        raw_citation=raw_citation,
        title=title or raw_citation,
        year=year,
        journal=journal,
        authors=authors,
        volume=volume,
        issue=issue,
        page=page,
        threshold=threshold,
        debug=debug,
    )
    oa = _run_openalex(
        title=title,
        year=year,
        journal=journal,
        authors=authors,
        mailto=mailto,
        debug=debug,
    )

    comp_score = _comparison_score(cr.get("cr_citation_title"), cr.get("cr_citation_raw"))

    row = {
        "raw_citation": _sanitize(raw_citation),
        "title": _sanitize(title),
        "api_raw_doi": cr.get("api_raw_doi"),
        "score_raw": cr.get("score_raw"),
        "valid_raw": cr.get("valid_raw"),
        "api_title_doi": cr.get("api_title_doi"),
        "score_title": cr.get("score_title"),
        "valid_title": cr.get("valid_title"),
        "oa_api_title_doi": oa.get("oa_api_title_doi"),
        "oa_relevancy_score": oa.get("oa_relevancy_score"),
        "oa_relevancy_score_code": oa.get("oa_relevancy_score_code"),
        "cr_citation_raw": _sanitize(cr.get("cr_citation_raw")),
        "cr_citation_title": _sanitize(cr.get("cr_citation_title")),
        "Comparison Score": comp_score,
    }
    return row


def main():
    ap = argparse.ArgumentParser(
        description="Export CrossRef+OpenAlex scoring evidence for references into a CSV."
    )
    ap.add_argument("--osf-id", required=True, help="OSF preprint id to process")
    ap.add_argument("--output", default="doi_evidence.csv", help="Output CSV path")
    ap.add_argument("--limit", type=int, default=400, help="Max references to fetch")
    ap.add_argument("--threshold", type=int, default=int(SBMV_THRESHOLD_DEFAULT), help="Crossref scoring threshold")
    ap.add_argument("--mailto", default=None, help="Override OPENALEX_MAILTO/OPENALEX_EMAIL for OpenAlex requests")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    mailto = args.mailto or os.environ.get("OPENALEX_MAILTO") or os.environ.get("OPENALEX_EMAIL") or OPENALEX_MAILTO

    repo = PreprintsRepo()
    refs = repo.select_refs_missing_doi(
        limit=args.limit,
        osf_id=args.osf_id,
        include_existing=True,
    )

    fields = [
        "raw_citation",
        "title",
        "api_raw_doi",
        "score_raw",
        "valid_raw",
        "api_title_doi",
        "score_title",
        "valid_title",
        "oa_api_title_doi",
        "oa_relevancy_score",
        "oa_relevancy_score_code",
        "cr_citation_raw",
        "cr_citation_title",
        "Comparison Score",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ref in refs:
            row = process_reference(ref, mailto, args.threshold, args.debug)
            writer.writerow(row)
            if args.debug:
                logger.info(json.dumps(row, ensure_ascii=False))

    print(f"CSV written to {args.output} with {len(refs)} rows")


if __name__ == "__main__":
    main()
