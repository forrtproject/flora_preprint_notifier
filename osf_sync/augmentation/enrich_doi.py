from __future__ import annotations
import time
from ..dynamo.preprints_repo import PreprintsRepo
from scripts.augmentation.matching_crossref import best_crossref_match  # reuse your logic
from scripts.augmentation.doi_check_openalex import match_title_and_year  # reuse your OpenAlex query

repo = PreprintsRepo()

def enrich_missing_with_crossref(limit: int = 200, sleep_seconds: float = 0.7) -> int:
    done = 0
    rows = repo.select_refs_missing_doi(limit=limit)
    for r in rows:
        entry = {
            "title": r["title"],
            "authors": r["authors"] or [],
            "journal": r["journal"],
            "year": r["year"],
        }
        match = best_crossref_match(entry)  # from your script
        if match and match.get("doi"):
            ok = repo.update_reference_doi(r["osf_id"], r["ref_id"], match["doi"], source="crossref")
            if ok:
                done += 1
        time.sleep(sleep_seconds)
    return done

def enrich_missing_with_openalex(limit: int = 200) -> int:
    done = 0
    rows = repo.select_refs_missing_doi(limit=limit)
    for r in rows:
        # Your OpenAlex function returns a list of candidates; pick first with DOI
        cands = match_title_and_year(r["title"], r["year"])
        doi = None
        for c in cands:
            if c.get("doi"):
                doi = c["doi"]
                break
        if doi:
            ok = repo.update_reference_doi(r["osf_id"], r["ref_id"], doi, source="openalex")
            if ok:
                done += 1
    return done
