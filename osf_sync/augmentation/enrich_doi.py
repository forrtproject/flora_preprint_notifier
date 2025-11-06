from __future__ import annotations
import time
from sqlalchemy import text
from ..db import engine
from scripts.augmentation.matching_crossref import best_crossref_match  # reuse your logic
from scripts.augmentation.doi_check_openalex import match_title_and_year  # reuse your OpenAlex query

SEL_MISSING = text("""
SELECT osf_id, ref_id, title, authors, journal, year
FROM preprint_references
WHERE doi IS NULL
ORDER BY osf_id, ref_id
LIMIT :limit
""")

UPD_CROSSREF = text("""
UPDATE preprint_references
SET doi = :doi,
    doi_source = 'crossref',
    doi_confidence = :conf,
    match_scores = :scores,
    updated_at = now()
WHERE osf_id = :osf_id AND ref_id = :ref_id
""")

UPD_OPENALEX = text("""
UPDATE preprint_references
SET doi = :doi,
    doi_source = 'openalex',
    updated_at = now()
WHERE osf_id = :osf_id AND ref_id = :ref_id AND doi IS NULL
""")

def enrich_missing_with_crossref(limit: int = 200, sleep_seconds: float = 0.7) -> int:
    done = 0
    with engine.begin() as conn:
        rows = conn.execute(SEL_MISSING, {"limit": limit}).mappings().all()
    for r in rows:
        entry = {
            "title": r["title"],
            "authors": r["authors"] or [],
            "journal": r["journal"],
            "year": r["year"],
        }
        match = best_crossref_match(entry)  # from your script
        if match and match.get("doi"):
            with engine.begin() as conn:
                conn.execute(UPD_CROSSREF, {
                    "doi": match["doi"],
                    "conf": match["confidence"],
                    "scores": match["scores"],
                    "osf_id": r["osf_id"],
                    "ref_id": r["ref_id"],
                })
            done += 1
        time.sleep(sleep_seconds)
    return done

def enrich_missing_with_openalex(limit: int = 200) -> int:
    done = 0
    with engine.begin() as conn:
        rows = conn.execute(SEL_MISSING, {"limit": limit}).mappings().all()
    for r in rows:
        # Your OpenAlex function returns a list of candidates; pick first with DOI
        cands = match_title_and_year(r["title"], r["year"])
        doi = None
        for c in cands:
            if c.get("doi"):
                doi = c["doi"]
                break
        if doi:
            with engine.begin() as conn:
                conn.execute(UPD_OPENALEX, {
                    "doi": doi,
                    "osf_id": r["osf_id"],
                    "ref_id": r["ref_id"],
                })
            done += 1
    return done