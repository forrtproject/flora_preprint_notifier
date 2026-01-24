from __future__ import annotations
from .matching_crossref import enrich_missing_with_crossref as _enrich_crossref
from .doi_check_openalex import enrich_missing_with_openalex as _enrich_openalex

def enrich_missing_with_crossref(limit: int = 200, sleep_seconds: float = 0.7) -> int:
    # Delegate to the updated Crossref pipeline (sleep_seconds unused; keep for compatibility).
    result = _enrich_crossref(limit=limit)
    return int(result.get("updated", 0))


def enrich_missing_with_openalex(limit: int = 200) -> int:
    # Delegate to the updated OpenAlex pipeline.
    result = _enrich_openalex(limit=limit)
    return int(result.get("updated", 0))
