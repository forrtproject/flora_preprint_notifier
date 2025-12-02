from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from ..dynamo.preprints_repo import PreprintsRepo
from .forrt_original_lookup import normalize_doi, lookup_originals_with_forrt
from ..logging_setup import get_logger, with_extras

logger = get_logger(__name__)


def _info(msg: str, **extras: Any) -> None:
    if extras:
        with_extras(logger, **extras).info(msg)
    else:
        logger.info(msg)


def _warn(msg: str, **extras: Any) -> None:
    if extras:
        with_extras(logger, **extras).warning(msg)
    else:
        logger.warning(msg)


def screen_forrt_replications(
    *,
    limit: int = 500,
    osf_id: Optional[str] = None,
    ref_id: Optional[str] = None,
    persist_flags: bool = True,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    For each preprint, look at references that have a FORRT original DOI mapping (replication -> original),
    check whether the original DOI is already cited in the same reference list. If not, flag the preprint
    as eligible for inclusion.
    """
    repo = PreprintsRepo()
    rows = repo.select_refs_with_forrt_original(
        limit=limit,
        osf_id=osf_id,
        ref_id=ref_id,
        include_missing_original=True,
    )

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        osfid = r.get("osf_id")
        if osfid:
            grouped[osfid].append(r)

    results: List[Dict[str, Any]] = []

    for pid, refs in grouped.items():
        # Collect all DOIs cited in this preprint (normalize)
        all_dois: Set[str] = set()
        for r in refs:
            d = normalize_doi(r.get("doi"))
            if d:
                all_dois.add(d)
        # Also consider other references without FORRT mapping (so fetch them too)
        try:
            extra = repo.select_refs_with_doi(limit=500, osf_id=pid, only_unchecked=False)
        except TypeError:
            extra = repo.select_refs_with_doi(limit=500, osf_id=pid)  # fallback for older signature
        for r in extra:
            d = normalize_doi(r.get("doi"))
            if d:
                all_dois.add(d)

        eligible_refs: List[Dict[str, Any]] = []
        retained_refs: List[Dict[str, Any]] = []

        for r in refs:
            # Original DOI is treated as the same as the replication DOI; field is no longer stored separately
            replication_doi = normalize_doi(r.get("doi"))
            orig = replication_doi
            refid = r.get("ref_id")
            already_cited = orig in all_dois if orig else False

            if persist_flags:
                try:
                    repo.update_reference_forrt_screening(pid, refid, original_cited=already_cited)
                except Exception as e:
                    _warn("Failed to persist FORRT screening flag", osf_id=pid, ref_id=refid, error=str(e))

            payload = {
                "osf_id": pid,
                "ref_id": refid,
                "replication_doi": replication_doi,
                "original_doi": orig,
                "original_cited": already_cited,
            }
            retained_refs.append(payload)
            if not already_cited:
                eligible_refs.append(payload)

        if eligible_refs:
            results.append({
                "osf_id": pid,
                "eligible": True,
                "eligible_count": len(eligible_refs),
                "replication_refs": retained_refs,
            })
        else:
            results.append({
                "osf_id": pid,
                "eligible": False,
                "eligible_count": 0,
                "replication_refs": retained_refs,
            })

        if debug:
            _info("FORRT screening", osf_id=pid, eligible_count=len(eligible_refs), total=len(retained_refs))

    return results


def lookup_and_screen_forrt(
    *,
    limit_lookup: int = 200,
    limit_screen: int = 500,
    osf_id: Optional[str] = None,
    ref_id: Optional[str] = None,
    cache_ttl_hours: Optional[int] = None,
    persist_flags: bool = True,
    only_unchecked: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper: first run FORRT lookup to populate original DOIs, then screen.
    Returns {"lookup": {...stats...}, "screen": [...results...]}.
    """
    lookup_stats = lookup_originals_with_forrt(
        limit=limit_lookup,
        osf_id=osf_id,
        ref_id=ref_id,
        only_unchecked=only_unchecked,
        cache_ttl_hours=cache_ttl_hours,
        debug=debug,
    )
    screen_results = screen_forrt_replications(
        limit=limit_screen,
        osf_id=osf_id,
        ref_id=ref_id,
        persist_flags=persist_flags,
        debug=debug,
    )
    return {"lookup": lookup_stats, "screen": screen_results}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Screen replication DOIs via FORRT original lookup comparison.")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--limit-lookup", type=int, default=200, help="How many rows to send to FORRT lookup before screening")
    ap.add_argument("--osf_id", default=None)
    ap.add_argument("--only-osf-id", dest="osf_id", default=None, help="Alias for --osf_id to process a single OSF id")
    ap.add_argument("--ref_id", default=None)
    ap.add_argument("--no-persist", action="store_true", help="Do not write screening flags back to Dynamo.")
    ap.add_argument("--no-lookup-first", action="store_true", help="Skip the lookup stage and only screen existing FORRT results")
    ap.add_argument("--include-checked", action="store_true", help="Re-run lookup even for rows with prior FORRT status")
    ap.add_argument("--cache-ttl-hours", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.no_lookup_first:
        out = screen_forrt_replications(
            limit=args.limit,
            osf_id=args.osf_id,
            ref_id=args.ref_id,
            persist_flags=not args.no_persist,
            debug=args.debug,
        )
        print(out)
    else:
        out = lookup_and_screen_forrt(
            limit_lookup=args.limit_lookup,
            limit_screen=args.limit,
            osf_id=args.osf_id,
            ref_id=args.ref_id,
            cache_ttl_hours=args.cache_ttl_hours,
            persist_flags=not args.no_persist,
            only_unchecked=not args.include_checked,
            debug=args.debug,
        )
        print(out)
