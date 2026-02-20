from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from ..dynamo.preprints_repo import PreprintsRepo
from .doi_multi_method_lookup import (
    DOI_MULTI_METHOD_CACHE_TTL_SECS,
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
    workers: int = 1,
) -> Dict[str, Any]:
    """
    Run the multi-method DOI pipeline for refs missing DOIs (or specific ref_id),
    and update Dynamo with the final match.
    """
    repo = PreprintsRepo()
    checked = updated = failed = 0
    use_threshold = int(threshold) if threshold is not None else int(
        STRUCTURED_THRESHOLD_DEFAULT)
    use_mailto = mailto or OPENALEX_MAILTO

    skip_checked = int(DOI_MULTI_METHOD_CACHE_TTL_SECS) if not (include_existing or bool(ref_id and osf_id)) else None
    include_existing_effective = include_existing or bool(ref_id and osf_id)
    rows: list[Dict[str, Any]]
    preprints_selected: Optional[int] = None
    limit_reached = False
    if osf_id or ref_id:
        rows = repo.select_refs_missing_doi(
            limit=limit,
            osf_id=osf_id,
            ref_id=ref_id,
            include_existing=include_existing_effective,
            skip_checked_within_seconds=skip_checked,
        )
    else:
        preprint_limit = max(1, int(limit))
        selected_osf_ids_all = repo.select_osf_ids_with_refs_missing_doi(
            limit_preprints=preprint_limit + 1,
            skip_checked_within_seconds=skip_checked,
        )
        selected_osf_ids = selected_osf_ids_all[:preprint_limit]
        preprints_selected = len(selected_osf_ids)
        limit_reached = len(selected_osf_ids_all) > preprint_limit
        rows = []
        for selected_osf_id in selected_osf_ids:
            rows.extend(
                repo.select_refs_missing_doi(
                    limit=None,
                    osf_id=selected_osf_id,
                    include_existing=include_existing_effective,
                    skip_checked_within_seconds=skip_checked,
                )
            )

    def _process_one(ref: Dict) -> Dict:
        row = process_reference(
            ref,
            threshold=use_threshold,
            mailto=use_mailto,
            screen_raw=screen_raw,
            debug=debug,
        )
        return {"ref": ref, "row": row}

    if workers > 1 and len(rows) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one, ref): ref for ref in rows}
            for future in as_completed(futures):
                checked += 1
                ref = futures[future]
                try:
                    result = future.result()
                    row = result["row"]
                    doi = _resolve_final_doi(row)
                    status = row.get("status")
                    if not doi or status != "matched":
                        repo.mark_reference_doi_checked(ref.get("osf_id"), ref.get("ref_id"))
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
                        extra={"osf_id": ref.get("osf_id"), "ref_id": ref.get("ref_id")},
                    )
    else:
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
                    repo.mark_reference_doi_checked(ref.get("osf_id"), ref.get("ref_id"))
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
                    extra={"osf_id": ref.get("osf_id"), "ref_id": ref.get("ref_id")},
                )

    out: Dict[str, Any] = {"checked": checked, "updated": updated, "failed": failed}
    if preprints_selected is not None:
        out["preprints_selected"] = preprints_selected
    out["limit_reached"] = limit_reached
    return out
