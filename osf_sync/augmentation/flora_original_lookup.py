from __future__ import annotations

import csv
import datetime as dt
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..dynamo.preprints_repo import PreprintsRepo
from ..logging_setup import get_logger, with_extras
from ..runtime_config import RUNTIME_CONFIG

logger = get_logger(__name__)

FLORA_CSV_URL = RUNTIME_CONFIG.flora.csv_url
FLORA_CSV_PATH_DEFAULT = RUNTIME_CONFIG.flora.csv_path


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


def _exception(msg: str, **extras: Any) -> None:
    if extras:
        with_extras(logger, **extras).exception(msg)
    else:
        logger.exception(msg)


_DOI_PATTERN = re.compile(r"10\.\S+", re.IGNORECASE)


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """
    Normalize DOI to lowercase and strip URL prefixes; return None if invalid.
    """
    if not doi or not isinstance(doi, str):
        return None
    v = doi.strip().lower()
    v = re.sub(r"^https?://(dx\.)?doi\.org/", "", v)
    m = _DOI_PATTERN.search(v)
    return m.group(0) if m else None


def _extract_ref_objects(payload: Any) -> List[Dict[str, Optional[str]]]:
    """
    Extract array of {doi_o, doi_r, apa_ref_o, apa_ref_r} objects from a payload.
    Used for backwards compatibility with legacy persisted payload rows.
    """
    out: List[Dict[str, Optional[str]]] = []

    def _append_pair(
        doi_o: Optional[str],
        doi_r: Optional[str],
        apa_ref_o: Optional[str],
        apa_ref_r: Optional[str],
        replication_outcome: Optional[str] = None,
    ) -> None:
        out.append(
            {
                "doi_o": normalize_doi(doi_o) if doi_o else None,
                "doi_r": normalize_doi(doi_r) if doi_r else None,
                "apa_ref_o": apa_ref_o,
                "apa_ref_r": apa_ref_r,
                "replication_outcome": replication_outcome,
            }
        )

    def _ensure_list(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        if isinstance(value, dict):
            return [value]
        return []

    def _from_record_lists(originals: Any, replications: Any) -> None:
        originals_list = _ensure_list(originals)
        replications_list = _ensure_list(replications)
        if originals_list and replications_list:
            for original in originals_list:
                for replication in replications_list:
                    _append_pair(
                        original.get("doi"),
                        replication.get("doi"),
                        original.get("apa_ref"),
                        replication.get("apa_ref"),
                        replication.get("outcome") or replication.get("replication_outcome"),
                    )
        else:
            for original in originals_list:
                _append_pair(
                    original.get("doi"),
                    None,
                    original.get("apa_ref"),
                    None,
                )
            for replication in replications_list:
                _append_pair(
                    None,
                    replication.get("doi"),
                    None,
                    replication.get("apa_ref"),
                )

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if any(k in obj for k in ("originals", "replications")):
                _from_record_lists(obj.get("originals"), obj.get("replications"))
            if any(k in obj for k in ("doi_o", "doi_r", "apa_ref_o", "apa_ref_r")):
                _append_pair(
                    obj.get("doi_o"),
                    obj.get("doi_r"),
                    obj.get("apa_ref_o"),
                    obj.get("apa_ref_r"),
                    obj.get("replication_outcome") or obj.get("outcome_r") or obj.get("outcome"),
                )
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(payload)

    seen: Dict[tuple, int] = {}
    uniq_out: List[Dict[str, Optional[str]]] = []
    for rec in out:
        key = (rec.get("doi_o"), rec.get("doi_r"), rec.get("apa_ref_o"), rec.get("apa_ref_r"))
        if key in seen:
            # Replace earlier record if this one has a non-empty outcome and the earlier didn't
            idx = seen[key]
            if rec.get("replication_outcome") and not uniq_out[idx].get("replication_outcome"):
                uniq_out[idx] = rec
            continue
        seen[key] = len(uniq_out)
        uniq_out.append(rec)
    return uniq_out


def _normalize_field_name(name: Optional[str]) -> str:
    return (name or "").replace("\ufeff", "").strip().strip('"').strip()


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _resolve_flora_csv_path(cache_path: Optional[str]) -> Path:
    raw = cache_path or FLORA_CSV_PATH_DEFAULT
    return Path(raw).expanduser()


def _is_file_from_today(path: Path) -> bool:
    try:
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
    except FileNotFoundError:
        return False
    except Exception:
        return False
    return mtime.date() >= dt.date.today()


def _download_flora_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with requests.get(FLORA_CSV_URL, stream=True, timeout=(20, 180)) as resp:
            resp.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _ensure_fresh_flora_csv(path: Path, *, debug: bool = False) -> Dict[str, Any]:
    if _is_file_from_today(path):
        return {"downloaded": False, "used_stale": False}

    try:
        _download_flora_csv(path)
        if debug:
            _info("Downloaded fresh FLORA CSV", path=str(path), source=FLORA_CSV_URL)
        return {"downloaded": True, "used_stale": False}
    except Exception as exc:
        if path.exists():
            _warn(
                "Failed to refresh FLORA CSV; using existing local copy",
                path=str(path),
                error=str(exc),
            )
            return {"downloaded": False, "used_stale": True}
        raise RuntimeError(f"Unable to download FLORA CSV to {path}") from exc


def _load_flora_pairs_by_original(path: Path) -> Dict[str, List[Dict[str, Optional[str]]]]:
    pairs_by_original: Dict[str, List[Dict[str, Optional[str]]]] = {}
    seen_by_original: Dict[str, set] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return pairs_by_original
        field_map = {_normalize_field_name(name): name for name in reader.fieldnames}

        def _row_value(row: Dict[str, Any], key: str) -> Optional[str]:
            return _clean_text(row.get(field_map.get(key, key)))

        for row in reader:
            doi_o = normalize_doi(_row_value(row, "doi_o"))
            if not doi_o:
                continue
            rec = {
                "doi_o": doi_o,
                "doi_r": normalize_doi(_row_value(row, "doi_r")),
                "apa_ref_o": _row_value(row, "apa_ref_o"),
                "apa_ref_r": _row_value(row, "apa_ref_r"),
            }
            key = (rec["doi_o"], rec["doi_r"], rec["apa_ref_o"], rec["apa_ref_r"])
            seen = seen_by_original.setdefault(doi_o, set())
            if key in seen:
                continue
            seen.add(key)
            pairs_by_original.setdefault(doi_o, []).append(rec)

    return pairs_by_original


def lookup_originals_with_flora(
    *,
    limit: int = 200,
    osf_id: Optional[str] = None,
    ref_id: Optional[str] = None,
    only_unchecked: bool = True,
    cache_path: Optional[str] = None,
    cache_ttl_hours: Optional[int] = None,
    ignore_cache: bool = False,
    debug: bool = False,
) -> Dict[str, int]:
    """
    Populate FLORA original/replication pairs from local flora.csv data.
    The file is refreshed once per day from FLORA's public CSV source.
    """
    repo = PreprintsRepo()
    rows = repo.select_refs_with_doi(limit=limit, osf_id=osf_id, ref_id=ref_id, only_unchecked=only_unchecked)
    if only_unchecked:
        # Treat status=False as terminal for local CSV mode; only process never-checked rows.
        rows = [r for r in rows if r and r.get("flora_lookup_status") is None]

    candidate_ids = sorted({(r or {}).get("osf_id") for r in rows if (r or {}).get("osf_id")})
    allowed_ids = repo.filter_osf_ids_without_sent_email(candidate_ids)
    filtered_rows = [r for r in rows if (r or {}).get("osf_id") in allowed_ids]
    skipped_sent_preprint = len(rows) - len(filtered_rows)
    rows = filtered_rows

    if cache_ttl_hours is not None:
        _warn("cache_ttl_hours is ignored for local FLORA CSV lookup", cache_ttl_hours=cache_ttl_hours)
    if ignore_cache:
        _warn("ignore_cache is ignored for local FLORA CSV lookup")

    flora_path = _resolve_flora_csv_path(cache_path)
    refresh_meta = _ensure_fresh_flora_csv(flora_path, debug=debug)
    flora_pairs = _load_flora_pairs_by_original(flora_path)

    stats: Dict[str, int] = {
        "checked": 0,
        "updated": 0,
        "failed": 0,
        "skipped_sent_preprint": skipped_sent_preprint,
        "cache_hits": 0,
        "csv_downloaded": 1 if refresh_meta.get("downloaded") else 0,
    }

    for r in rows:
        osfid = r.get("osf_id")
        refid = r.get("ref_id")
        doi = normalize_doi(r.get("doi"))
        if not doi:
            continue

        stats["checked"] += 1
        ref_pairs = flora_pairs.get(doi) or []
        status = bool(ref_pairs)
        try:
            repo.update_reference_flora(
                osfid,
                refid,
                status=status,
                ref_pairs=ref_pairs,
            )
            stats["updated"] += 1
        except Exception:
            stats["failed"] += 1
            _exception(
                "Failed to update FLORA lookup result",
                osf_id=osfid,
                ref_id=refid,
                match_found=status,
            )

    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Lookup originals via local FLORA CSV for references that already have DOIs."
    )
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--osf_id", default=None)
    ap.add_argument("--ref_id", default=None)
    ap.add_argument("--no-only-unchecked", action="store_true", help="Process all DOI rows even if already checked.")
    ap.add_argument(
        "--cache-path",
        default=None,
        help="Override FLORA CSV path (defaults to flora.csv_path in config/runtime.toml).",
    )
    ap.add_argument(
        "--cache-ttl-hours",
        type=int,
        default=None,
        help="Deprecated; ignored when using local FLORA CSV lookup.",
    )
    ap.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Deprecated; ignored when using local FLORA CSV lookup.",
    )
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out = lookup_originals_with_flora(
        limit=args.limit,
        osf_id=args.osf_id,
        ref_id=args.ref_id,
        only_unchecked=not args.no_only_unchecked,
        cache_path=args.cache_path,
        cache_ttl_hours=args.cache_ttl_hours,
        ignore_cache=args.ignore_cache,
        debug=args.debug,
    )
    print(out)
