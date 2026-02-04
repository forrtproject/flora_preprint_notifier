from __future__ import annotations

import logging
from typing import Dict, Optional

from ..dynamo.preprints_repo import PreprintsRepo
from scripts.manual_post_grobid.doi_multi_method_lookup import (
    OPENALEX_MAILTO,
    STRUCTURED_THRESHOLD_DEFAULT,
    process_reference,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logger.setLevel(logging.INFO)


def _resolve_final_doi(row: Dict[str, object]) -> Optional[str]:
    method = row.get("final_method")
    if method == "crossref_raw":
        return row.get("raw_citation_doi")  # type: ignore[return-value]
    if method == "crossref_title":
        return row.get("title_doi")  # type: ignore[return-value]
    if method == "openalex_title":
        return row.get("openalex_doi")  # type: ignore[return-value]
    return None


def _resolve_source(method: Optional[str]) -> str:
    if not method:
        return "multi_method"
    if str(method).startswith("crossref"):
        return "crossref"
    if str(method).startswith("openalex"):
        return "openalex"
    return "multi_method"


def enrich_missing_with_multi_method(
    *,
    limit: int = 300,
    threshold: Optional[int] = None,
    mailto: Optional[str] = None,
    osf_id: Optional[str] = None,
    ref_id: Optional[str] = None,
    include_existing: bool = False,
    screen_raw: bool = True,
    debug: bool = False,
) -> Dict[str, int]:
    """
    Run the multi-method DOI pipeline for refs missing DOIs (or specific ref_id),
    and update Dynamo with the final match.
    """
    repo = PreprintsRepo()
    checked = updated = failed = 0
    use_threshold = int(threshold) if threshold is not None else int(
        STRUCTURED_THRESHOLD_DEFAULT)
    use_mailto = mailto or OPENALEX_MAILTO

    rows = repo.select_refs_missing_doi(
        limit=limit,
        osf_id=osf_id,
        ref_id=ref_id,
        include_existing=include_existing or bool(ref_id and osf_id),
    )

    for ref in rows:
        checked += 1
        try:
            row = process_reference(
                ref,
                threshold=use_threshold,
                mailto=use_mailto,
                screen_raw=screen_raw,
                debug=debug,
            )
            doi = _resolve_final_doi(row)
            status = row.get("status")
            if not doi or status != "matched":
                continue
            source = _resolve_source(row.get("final_method"))
            ok = repo.update_reference_doi(
                ref.get("osf_id"), ref.get("ref_id"), doi, source=source)
            if ok:
                updated += 1
        except Exception:
            failed += 1
            logger.exception(
                "Multi-method enrichment failed",
                extra={"osf_id": ref.get(
                    "osf_id"), "ref_id": ref.get("ref_id")},
            )

    return {"checked": checked, "updated": updated, "failed": failed}
