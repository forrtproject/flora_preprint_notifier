from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from ..dynamo.preprints_repo import PreprintsRepo
from ..dynamo.api_cache_repo import ApiCacheRepo
from .forrt_original_lookup import normalize_doi, lookup_originals_with_forrt, _extract_ref_objects, _cache_key_for_doi
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
    For each preprint, match the preprint's own DOI to FORRT originals (doi_o) and see whether
    any paired replication DOI (doi_r) is already cited in the same reference list. If a replication DOI
    is present, mark the reference as cited.
    """
    repo = PreprintsRepo()
    cache_repo = ApiCacheRepo()
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
        preprint_doi = normalize_doi(repo.get_preprint_doi(pid))

        # Collect all DOIs cited in this preprint (normalize)
        all_dois: Set[str] = set()
        for r in refs:
            d = normalize_doi(r.get("doi"))
            if d:
                all_dois.add(d)
        # Also consider other references without FORRT mapping (so fetch them too)
        try:
            extra = repo.select_refs_with_doi(
                limit=500, osf_id=pid, only_unchecked=False)
        except TypeError:
            extra = repo.select_refs_with_doi(
                limit=500, osf_id=pid)  # fallback for older signature
        for r in extra:
            d = normalize_doi(r.get("doi"))
            if d:
                all_dois.add(d)

        eligible_refs: List[Dict[str, Any]] = []
        retained_refs: List[Dict[str, Any]] = []

        for r in refs:
            refid = r.get("ref_id")

            # Filter FORRT ref objects so we only consider pairs whose doi_o matches the preprint DOI
            ref_objs = r.get("forrt_ref_pairs") or []
            if not ref_objs:
                payload = r.get("forrt_lookup_payload")
                if payload:
                    ref_objs = _extract_ref_objects(payload)
                else:
                    doi_key = normalize_doi(r.get("doi"))
                    if doi_key:
                        cached = cache_repo.get(_cache_key_for_doi(doi_key))
                        cached_payload = cached.get(
                            "payload") if cached else None
                        if cached_payload:
                            ref_objs = _extract_ref_objects(cached_payload)
            matching_pairs = []
            for obj in ref_objs:
                doi_o = normalize_doi(obj.get("doi_o"))
                doi_r = normalize_doi(obj.get("doi_r"))
                if preprint_doi and doi_o and preprint_doi == doi_o:
                    matching_pairs.append({"doi_o": doi_o, "doi_r": doi_r})

            replication_dois = [p["doi_r"]
                                for p in matching_pairs if p.get("doi_r")]
            replication_cited = any(
                (d in all_dois) for d in replication_dois) if replication_dois else False

            if persist_flags:
                try:
                    repo.update_reference_forrt_screening(
                        pid,
                        refid,
                        original_cited=replication_cited,
                    )
                except Exception as e:
                    _warn("Failed to persist FORRT screening flag",
                          osf_id=pid, ref_id=refid, error=str(e))

            payload = {
                "osf_id": pid,
                "ref_id": refid,
                "original_doi": preprint_doi,
                "matching_replication_dois": replication_dois,
                "replication_cited": replication_cited,
            }
            retained_refs.append(payload)
            if not replication_cited:
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
            _info("FORRT screening", osf_id=pid, eligible_count=len(
                eligible_refs), total=len(retained_refs))

    return results


def lookup_and_screen_forrt(
    *,
    limit_lookup: int = 200,
    limit_screen: int = 500,
    osf_id: Optional[str] = None,
    ref_id: Optional[str] = None,
    cache_ttl_hours: Optional[int] = None,
    ignore_cache: bool = False,
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
        ignore_cache=ignore_cache,
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
    ap = argparse.ArgumentParser(
        description="Screen replication DOIs via FORRT original lookup comparison.")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--limit-lookup", type=int, default=200,
                    help="How many rows to send to FORRT lookup before screening")
    ap.add_argument("--osf_id", default=None)
    ap.add_argument("--only-osf-id", dest="osf_id", default=None,
                    help="Alias for --osf_id to process a single OSF id")
    ap.add_argument("--ref_id", default=None)
    ap.add_argument("--no-persist", action="store_true",
                    help="Do not write screening flags back to Dynamo.")
    ap.add_argument("--no-lookup-first", action="store_true",
                    help="Skip the lookup stage and only screen existing FORRT results")
    ap.add_argument("--include-checked", action="store_true",
                    help="Re-run lookup even for rows with prior FORRT status")
    ap.add_argument("--cache-ttl-hours", type=int, default=None)
    ap.add_argument("--ignore-cache", action="store_true",
                    help="Bypass database cache and call FORRT again")
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
            ignore_cache=args.ignore_cache,
            persist_flags=not args.no_persist,
            only_unchecked=not args.include_checked,
            debug=args.debug,
        )
        print(out)
