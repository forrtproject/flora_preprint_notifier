from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import socket
import time
import uuid
from typing import Any, Dict, Optional

import requests

from .augmentation.doi_multi_method import enrich_missing_with_multi_method
from .augmentation.forrt_screening import lookup_and_screen_forrt
from .augmentation.run_extract import extract_for_osf_id
from .author_randomization import run_author_randomization
from .db import init_db
from .dynamo.preprints_repo import PreprintsRepo
from .fetch_one import fetch_preprint_by_doi, fetch_preprint_by_id, upsert_one_preprint
from .grobid import mark_tei, process_pdf_to_tei
from .iter_preprints import iter_preprints_batches, iter_preprints_range
from .pdf import ensure_pdf_available_or_delete, mark_downloaded
from .upsert import upsert_batch
from .extraction.extract_author_list import run_author_extract

OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", "you@example.com")
PDF_DEST_ROOT = os.environ.get("PDF_DEST_ROOT", "/data/preprints")
SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL")
SOURCE_KEY_ALL = "osf:all"
DEFAULT_LEASE_SECONDS = int(os.environ.get("PIPELINE_CLAIM_LEASE_SECONDS", "1800"))

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logger.setLevel(logging.INFO)


def _slack(msg: str, *, level: str = "info", extra: Optional[Dict[str, Any]] = None) -> None:
    if not SLACK_WEBHOOK:
        return
    payload = {"text": f"*[{level.upper()}]* {msg}"}
    if extra:
        payload["attachments"] = [
            {"text": "```" + json.dumps(extra, ensure_ascii=False, indent=2) + "```"}
        ]
    try:
        requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
    except Exception:
        logger.debug("Slack send failed", exc_info=True)


def _parse_iso_dt(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        v = value.replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(v)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        else:
            d = d.astimezone(dt.timezone.utc)
        return d
    except Exception:
        return None


def _get_cursor(source_key: str) -> Optional[dt.datetime]:
    repo = PreprintsRepo()
    v = repo.get_cursor(source_key)
    if not v:
        return None
    try:
        vv = v.replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(vv)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d
    except Exception:
        return None


def _set_cursor(source_key: str, last_seen: dt.datetime) -> None:
    repo = PreprintsRepo()
    repo.set_cursor(source_key, last_seen.isoformat())


def _make_owner(owner: Optional[str] = None) -> str:
    if owner:
        return owner
    host = socket.gethostname()
    pid = os.getpid()
    return f"{host}:{pid}:{uuid.uuid4().hex[:8]}"


def _deadline(max_seconds: Optional[int]) -> Optional[float]:
    if max_seconds is None or max_seconds <= 0:
        return None
    return time.monotonic() + max_seconds


def _time_up(deadline: Optional[float]) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def sync_from_osf(
    *,
    subject_text: Optional[str] = None,
    batch_size: int = 1000,
    limit: Optional[int] = None,
    max_seconds: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    init_db()
    source_key = f"osf:{subject_text}" if subject_text else SOURCE_KEY_ALL
    since_dt = _get_cursor(source_key)
    if since_dt is None:
        since_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
    since_iso_date = since_dt.astimezone(dt.timezone.utc).date().isoformat()

    total_upserted = 0
    processed = 0
    max_published_seen: Optional[dt.datetime] = None
    deadline = _deadline(max_seconds)

    logger.info("sync_from_osf start", extra={"since": since_iso_date, "subject": subject_text})

    for batch in iter_preprints_batches(
        since_date=since_iso_date,
        subject_text=subject_text,
        batch_size=batch_size,
        sort="date_published",
    ):
        if _time_up(deadline):
            break

        effective_batch = batch
        if limit is not None:
            remaining = limit - processed
            if remaining <= 0:
                break
            effective_batch = batch[:remaining]

        for obj in effective_batch:
            pub = _parse_iso_dt((obj.get("attributes") or {}).get("date_published"))
            if pub and (max_published_seen is None or pub > max_published_seen):
                max_published_seen = pub

        processed += len(effective_batch)
        if not dry_run:
            total_upserted += upsert_batch(effective_batch)
        else:
            total_upserted += len(effective_batch)

        logger.info("upserted batch", extra={"batch_size": len(effective_batch), "total": total_upserted})
        time.sleep(0.2)

    cursor_out = (max_published_seen or since_dt).isoformat()
    if max_published_seen and not dry_run:
        _set_cursor(source_key, max_published_seen)

    out = {
        "upserted": total_upserted,
        "cursor": cursor_out,
        "dry_run": dry_run,
        "stopped_due_to_time": _time_up(deadline),
    }
    _slack("OSF sync finished", extra=out)
    return out


def sync_from_date_to_now(
    *,
    start_date: str,
    subject_text: Optional[str] = None,
    batch_size: int = 1000,
    limit: Optional[int] = None,
    max_seconds: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    init_db()
    total = 0
    processed = 0
    deadline = _deadline(max_seconds)
    logger.info("sync_from_date_to_now", extra={"start": start_date, "subject": subject_text})

    for batch in iter_preprints_range(
        start_date=start_date,
        until_date=None,
        subject_text=subject_text,
        batch_size=batch_size,
    ):
        if _time_up(deadline):
            break

        effective_batch = batch
        if limit is not None:
            remaining = limit - processed
            if remaining <= 0:
                break
            effective_batch = batch[:remaining]

        processed += len(effective_batch)
        if not dry_run:
            total += upsert_batch(effective_batch)
        else:
            total += len(effective_batch)
        logger.info("upserted batch", extra={"batch_size": len(effective_batch), "total": total})

    out = {
        "upserted": total,
        "from": start_date,
        "to": "now",
        "dry_run": dry_run,
        "stopped_due_to_time": _time_up(deadline),
    }
    _slack("Ad-hoc OSF window sync finished", extra=out)
    return out


def download_single_pdf(osf_id: str) -> Dict[str, Any]:
    repo = PreprintsRepo()
    row = repo.get_preprint_basic(osf_id)
    if not row:
        return {"osf_id": osf_id, "skipped": "no longer in DB"}

    provider_id = row["provider_id"] or "unknown"

    kind, path = ensure_pdf_available_or_delete(
        osf_id=row["osf_id"],
        provider_id=provider_id,
        raw=row["raw"],
        dest_root=PDF_DEST_ROOT,
    )

    if kind == "deleted":
        return {"osf_id": osf_id, "deleted": True, "reason": "unsupported file type"}

    mark_downloaded(osf_id=row["osf_id"], local_path=path, ok=True)
    logger.info("PDF saved", extra={"osf_id": osf_id, "path": path})
    return {"osf_id": osf_id, "downloaded": True, "source": kind, "path": path}


def grobid_single(osf_id: str) -> Dict[str, Any]:
    repo = PreprintsRepo()
    row = repo.get_preprint_basic(osf_id)
    if not row:
        return {"osf_id": osf_id, "skipped": "not found"}

    full = repo.t_preprints.get_item(Key={"osf_id": osf_id}).get("Item") or {}
    if not full.get("pdf_downloaded"):
        return {"osf_id": osf_id, "skipped": "pdf not downloaded"}
    if full.get("tei_generated"):
        return {"osf_id": osf_id, "skipped": "already processed"}

    provider_id = row["provider_id"] or "unknown"
    ok, tei_path, err = process_pdf_to_tei(provider_id, osf_id)
    if ok:
        mark_tei(osf_id, ok=True, tei_path=tei_path)
    logger.info("GROBID done", extra={"osf_id": osf_id, "ok": ok, "tei": tei_path})
    return {"osf_id": osf_id, "ok": ok, "tei_path": tei_path, "error": err}


def extract_from_tei(provider_id: str, osf_id: str) -> Dict[str, Any]:
    base = os.environ.get("PDF_DEST_ROOT", PDF_DEST_ROOT)
    summary = extract_for_osf_id(provider_id, osf_id, base)
    return {
        "osf_id": osf_id,
        "parsed_ok": bool(summary.get("parsed_ok")),
        "written_ok": bool(summary.get("written_ok")),
        "references_upserted": int(summary.get("refs_count") or 0),
        "error": summary.get("error"),
    }


def process_pdf_batch(
    *,
    limit: int = 100,
    owner: Optional[str] = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    max_seconds: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    repo = PreprintsRepo()
    owner_id = _make_owner(owner)
    deadline = _deadline(max_seconds)

    candidates = repo.select_for_pdf(limit=max(limit * 3, limit))
    claimed = processed = failed = skipped_claimed = 0

    for osf_id in candidates:
        if processed + failed >= limit or _time_up(deadline):
            break
        if not repo.claim_stage_item("pdf", osf_id, owner=owner_id, lease_seconds=lease_seconds):
            skipped_claimed += 1
            continue

        claimed += 1
        if dry_run:
            repo.release_stage_claim("pdf", osf_id)
            processed += 1
            continue

        try:
            download_single_pdf(osf_id)
            processed += 1
        except Exception as exc:
            failed += 1
            repo.record_stage_error("pdf", osf_id, str(exc))
            logger.exception("PDF stage failed", extra={"osf_id": osf_id})

    out = {
        "stage": "pdf",
        "owner": owner_id,
        "selected": len(candidates),
        "claimed": claimed,
        "processed": processed,
        "failed": failed,
        "skipped_claimed": skipped_claimed,
        "dry_run": dry_run,
        "stopped_due_to_time": _time_up(deadline),
    }
    _slack("PDF stage finished", extra=out)
    return out


def process_grobid_batch(
    *,
    limit: int = 50,
    owner: Optional[str] = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    max_seconds: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    repo = PreprintsRepo()
    owner_id = _make_owner(owner)
    deadline = _deadline(max_seconds)

    candidates = repo.select_for_grobid(limit=max(limit * 3, limit))
    claimed = processed = failed = skipped_claimed = 0

    for osf_id in candidates:
        if processed + failed >= limit or _time_up(deadline):
            break
        if not repo.claim_stage_item("grobid", osf_id, owner=owner_id, lease_seconds=lease_seconds):
            skipped_claimed += 1
            continue

        claimed += 1
        if dry_run:
            repo.release_stage_claim("grobid", osf_id)
            processed += 1
            continue

        try:
            result = grobid_single(osf_id)
            if result.get("skipped"):
                repo.release_stage_claim("grobid", osf_id)
                processed += 1
            elif not result.get("ok"):
                failed += 1
                repo.record_stage_error("grobid", osf_id, str(result.get("error") or "unknown error"))
            else:
                processed += 1
        except Exception as exc:
            failed += 1
            repo.record_stage_error("grobid", osf_id, str(exc))
            logger.exception("GROBID stage failed", extra={"osf_id": osf_id})

    out = {
        "stage": "grobid",
        "owner": owner_id,
        "selected": len(candidates),
        "claimed": claimed,
        "processed": processed,
        "failed": failed,
        "skipped_claimed": skipped_claimed,
        "dry_run": dry_run,
        "stopped_due_to_time": _time_up(deadline),
    }
    _slack("GROBID stage finished", extra=out)
    return out


def process_extract_batch(
    *,
    limit: int = 200,
    owner: Optional[str] = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    max_seconds: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    repo = PreprintsRepo()
    owner_id = _make_owner(owner)
    deadline = _deadline(max_seconds)

    candidates = repo.select_for_extraction(limit=max(limit * 3, limit))
    claimed = processed = failed = skipped_claimed = 0

    for item in candidates:
        if processed + failed >= limit or _time_up(deadline):
            break

        osf_id = item.get("osf_id")
        if not osf_id:
            continue

        if not repo.claim_stage_item("extract", osf_id, owner=owner_id, lease_seconds=lease_seconds):
            skipped_claimed += 1
            continue

        claimed += 1
        if dry_run:
            repo.release_stage_claim("extract", osf_id)
            processed += 1
            continue

        provider_id = item.get("provider_id") or "unknown"
        try:
            result = extract_from_tei(provider_id, osf_id)
            if result.get("written_ok"):
                processed += 1
            else:
                failed += 1
                repo.record_stage_error("extract", osf_id, str(result.get("error") or "extract failed"))
        except Exception as exc:
            failed += 1
            repo.record_stage_error("extract", osf_id, str(exc))
            logger.exception("Extract stage failed", extra={"osf_id": osf_id})

    out = {
        "stage": "extract",
        "owner": owner_id,
        "selected": len(candidates),
        "claimed": claimed,
        "processed": processed,
        "failed": failed,
        "skipped_claimed": skipped_claimed,
        "dry_run": dry_run,
        "stopped_due_to_time": _time_up(deadline),
    }
    _slack("TEI extract stage finished", extra=out)
    return out


def process_enrich_batch(
    *,
    limit: int = 300,
    threshold: Optional[int] = None,
    mailto: Optional[str] = OPENALEX_EMAIL,
    osf_id: Optional[str] = None,
    ref_id: Optional[str] = None,
    debug: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {
            "stage": "enrich",
            "checked": 0,
            "updated": 0,
            "failed": 0,
            "dry_run": True,
        }
    stats = enrich_missing_with_multi_method(
        limit=limit,
        threshold=threshold,
        mailto=mailto,
        osf_id=osf_id,
        ref_id=ref_id,
        debug=debug,
    )
    out = {"stage": "enrich", **stats, "dry_run": False}
    _slack("Reference enrichment finished", extra=out)
    return out


def process_forrt_batch(
    *,
    limit_lookup: int = 200,
    limit_screen: int = 500,
    osf_id: Optional[str] = None,
    ref_id: Optional[str] = None,
    cache_ttl_hours: Optional[int] = None,
    persist_flags: bool = True,
    only_unchecked: bool = True,
    debug: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {
            "stage": "forrt",
            "lookup": {"checked": 0, "updated": 0, "failed": 0},
            "screen": [],
            "dry_run": True,
        }
    out = lookup_and_screen_forrt(
        limit_lookup=limit_lookup,
        limit_screen=limit_screen,
        osf_id=osf_id,
        ref_id=ref_id,
        cache_ttl_hours=cache_ttl_hours,
        persist_flags=persist_flags,
        only_unchecked=only_unchecked,
        debug=debug,
    )
    result = {"stage": "forrt", **out, "dry_run": False}
    _slack(
        "FORRT lookup/screen finished",
        extra={"lookup": out.get("lookup", {}), "screen_count": len(out.get("screen", []))},
    )
    return result


def process_author_batch(
    *,
    osf_ids: Optional[list[str]] = None,
    ids_file: Optional[str] = None,
    limit: Optional[int] = None,
    out: Optional[str] = None,
    pdf_root: Optional[str] = None,
    keep_files: bool = False,
    debug: bool = False,
    debug_log: Optional[str] = None,
    match_emails_file: Optional[str] = None,
    match_emails_threshold: float = 0.90,
    include_existing: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {"stage": "author", "exit_code": 0, "dry_run": True}
    code = run_author_extract(
        osf_ids=osf_ids,
        ids_file=ids_file,
        limit=limit,
        out=out,
        pdf_root=pdf_root,
        keep_files=keep_files,
        debug=debug,
        debug_log=debug_log,
        match_emails_file=match_emails_file,
        match_emails_threshold=match_emails_threshold,
        include_existing=include_existing,
    )
    out = {"stage": "author", "exit_code": code, "dry_run": False}
    _slack("Author extraction finished", extra=out)
    return out


def sync_one_by_id(*, osf_id: str, run_pdf_and_grobid: bool = True, run_extract: bool = True) -> Dict[str, Any]:
    data = fetch_preprint_by_id(osf_id)
    if not data:
        return {"ok": False, "reason": "not found", "osf_id": osf_id}

    upserted = upsert_one_preprint(data)
    result: Dict[str, Any] = {"ok": True, "osf_id": data["id"], "upserted": upserted}

    if run_pdf_and_grobid:
        pdf = download_single_pdf(data["id"])
        result["pdf"] = pdf
        grobid = grobid_single(data["id"])
        result["grobid"] = grobid
        if run_extract and grobid.get("ok"):
            repo = PreprintsRepo()
            basic = repo.get_preprint_basic(data["id"])
            provider_id = (basic or {}).get("provider_id") or "unknown"
            result["extract"] = extract_from_tei(provider_id, data["id"])
    return result


def sync_one_by_doi(*, doi_or_url: str, run_pdf_and_grobid: bool = True, run_extract: bool = True) -> Dict[str, Any]:
    data = fetch_preprint_by_doi(doi_or_url)
    if not data:
        return {"ok": False, "reason": "not found", "doi": doi_or_url}

    upserted = upsert_one_preprint(data)
    result: Dict[str, Any] = {"ok": True, "osf_id": data["id"], "upserted": upserted}

    if run_pdf_and_grobid:
        pdf = download_single_pdf(data["id"])
        result["pdf"] = pdf
        grobid = grobid_single(data["id"])
        result["grobid"] = grobid
        if run_extract and grobid.get("ok"):
            repo = PreprintsRepo()
            basic = repo.get_preprint_basic(data["id"])
            provider_id = (basic or {}).get("provider_id") or "unknown"
            result["extract"] = extract_from_tei(provider_id, data["id"])
    return result


def run_stage(args: argparse.Namespace) -> Dict[str, Any]:
    stage = args.stage
    if stage == "sync":
        return sync_from_osf(
            subject_text=args.subject,
            batch_size=args.batch_size,
            limit=args.limit,
            max_seconds=args.max_seconds,
            dry_run=args.dry_run,
        )
    if stage == "pdf":
        return process_pdf_batch(
            limit=args.limit or 100,
            owner=args.owner,
            lease_seconds=args.lease_seconds,
            max_seconds=args.max_seconds,
            dry_run=args.dry_run,
        )
    if stage == "grobid":
        return process_grobid_batch(
            limit=args.limit or 50,
            owner=args.owner,
            lease_seconds=args.lease_seconds,
            max_seconds=args.max_seconds,
            dry_run=args.dry_run,
        )
    if stage == "extract":
        return process_extract_batch(
            limit=args.limit or 200,
            owner=args.owner,
            lease_seconds=args.lease_seconds,
            max_seconds=args.max_seconds,
            dry_run=args.dry_run,
        )
    if stage == "enrich":
        return process_enrich_batch(
            limit=args.limit or 300,
            threshold=args.threshold,
            mailto=args.mailto,
            osf_id=args.osf_id,
            ref_id=args.ref_id,
            debug=args.debug,
            dry_run=args.dry_run,
        )
    if stage == "forrt":
        return process_forrt_batch(
            limit_lookup=args.limit_lookup,
            limit_screen=args.limit_screen,
            osf_id=args.osf_id,
            ref_id=args.ref_id,
            cache_ttl_hours=args.cache_ttl_hours,
            persist_flags=not args.no_persist,
            only_unchecked=not args.include_checked,
            debug=args.debug,
            dry_run=args.dry_run,
        )
    if stage == "author":
        return process_author_batch(
            osf_ids=args.author_osf_ids,
            ids_file=args.ids_file,
            limit=args.limit,
            out=args.out,
            pdf_root=args.pdf_root,
            keep_files=args.keep_files,
            debug=args.debug,
            debug_log=args.debug_log,
            match_emails_file=args.match_emails_file,
            match_emails_threshold=args.match_emails_threshold,
            include_existing=args.include_existing,
            dry_run=args.dry_run,
        )
    raise ValueError(f"Unsupported stage: {stage}")


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {"stages": {}}

    out["stages"]["sync"] = sync_from_osf(
        subject_text=args.subject,
        batch_size=args.batch_size,
        limit=args.sync_limit,
        max_seconds=args.max_seconds_per_stage,
        dry_run=args.dry_run,
    )
    out["stages"]["pdf"] = process_pdf_batch(
        limit=args.pdf_limit,
        owner=args.owner,
        lease_seconds=args.lease_seconds,
        max_seconds=args.max_seconds_per_stage,
        dry_run=args.dry_run,
    )
    out["stages"]["grobid"] = process_grobid_batch(
        limit=args.grobid_limit,
        owner=args.owner,
        lease_seconds=args.lease_seconds,
        max_seconds=args.max_seconds_per_stage,
        dry_run=args.dry_run,
    )
    out["stages"]["extract"] = process_extract_batch(
        limit=args.extract_limit,
        owner=args.owner,
        lease_seconds=args.lease_seconds,
        max_seconds=args.max_seconds_per_stage,
        dry_run=args.dry_run,
    )
    out["stages"]["enrich"] = process_enrich_batch(
        limit=args.enrich_limit,
        threshold=args.threshold,
        mailto=args.mailto,
        osf_id=args.osf_id,
        ref_id=args.ref_id,
        debug=args.debug,
        dry_run=args.dry_run,
    )
    out["stages"]["forrt"] = process_forrt_batch(
        limit_lookup=args.limit_lookup,
        limit_screen=args.limit_screen,
        osf_id=args.osf_id,
        ref_id=args.ref_id,
        cache_ttl_hours=args.cache_ttl_hours,
        persist_flags=not args.no_persist,
        only_unchecked=not args.include_checked,
        debug=args.debug,
        dry_run=args.dry_run,
    )

    if not args.skip_author:
        out["stages"]["author"] = process_author_batch(
            osf_ids=args.author_osf_ids,
            ids_file=args.ids_file,
            limit=args.author_limit,
            out=args.out,
            pdf_root=args.pdf_root,
            keep_files=not args.cleanup_author_files,
            debug=args.debug,
            debug_log=args.debug_log,
            match_emails_file=args.match_emails_file,
            match_emails_threshold=args.match_emails_threshold,
            include_existing=args.include_existing,
            dry_run=args.dry_run,
        )

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run OSF pipeline stages without Celery")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a single pipeline stage")
    p_run.add_argument("--stage", required=True, choices=["sync", "pdf", "grobid", "extract", "enrich", "forrt", "author"])
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--max-seconds", type=int, default=None)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--debug", action="store_true")
    p_run.add_argument("--batch-size", type=int, default=1000)
    p_run.add_argument("--subject", default=None)
    p_run.add_argument("--owner", default=None)
    p_run.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    p_run.add_argument("--threshold", type=int, default=None)
    p_run.add_argument("--mailto", default=OPENALEX_EMAIL)
    p_run.add_argument("--osf-id", default=None, help="Restrict enrich/FORRT to a specific OSF id")
    p_run.add_argument("--author-osf-id", action="append", dest="author_osf_ids", default=[])
    p_run.add_argument("--ids-file", default=None)
    p_run.add_argument("--out", default=None)
    p_run.add_argument("--pdf-root", default=None)
    p_run.add_argument("--keep-files", action="store_true")
    p_run.add_argument("--debug-log", default=None)
    p_run.add_argument("--match-emails-file", default=None)
    p_run.add_argument("--match-emails-threshold", type=float, default=0.90)
    p_run.add_argument("--include-existing", action="store_true")
    p_run.add_argument("--ref-id", default=None)
    p_run.add_argument("--limit-lookup", type=int, default=200)
    p_run.add_argument("--limit-screen", type=int, default=500)
    p_run.add_argument("--cache-ttl-hours", type=int, default=None)
    p_run.add_argument("--no-persist", action="store_true")
    p_run.add_argument("--include-checked", action="store_true")
    p_run.set_defaults(func=run_stage)

    p_all = sub.add_parser("run-all", help="Run the full pipeline (bounded per stage)")
    p_all.add_argument("--sync-limit", type=int, default=1000)
    p_all.add_argument("--pdf-limit", type=int, default=100)
    p_all.add_argument("--grobid-limit", type=int, default=50)
    p_all.add_argument("--extract-limit", type=int, default=200)
    p_all.add_argument("--enrich-limit", type=int, default=300)
    p_all.add_argument("--author-limit", type=int, default=None)
    p_all.add_argument("--limit-lookup", type=int, default=200)
    p_all.add_argument("--limit-screen", type=int, default=500)
    p_all.add_argument("--max-seconds-per-stage", type=int, default=None)
    p_all.add_argument("--skip-author", action="store_true", help="Skip author extraction stage")
    p_all.add_argument("--subject", default=None)
    p_all.add_argument("--batch-size", type=int, default=1000)
    p_all.add_argument("--threshold", type=int, default=None)
    p_all.add_argument("--mailto", default=OPENALEX_EMAIL)
    p_all.add_argument("--owner", default=None)
    p_all.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    p_all.add_argument("--osf-id", default=None, help="Restrict enrich/FORRT to a specific OSF id")
    p_all.add_argument("--author-osf-id", action="append", dest="author_osf_ids", default=[])
    p_all.add_argument("--ids-file", default=None)
    p_all.add_argument("--out", default=None)
    p_all.add_argument("--pdf-root", default=None)
    p_all.add_argument(
        "--cleanup-author-files",
        action="store_true",
        help="Allow author stage to delete local PDF/TEI files after processing (default is keep files)",
    )
    p_all.add_argument("--debug", action="store_true")
    p_all.add_argument("--debug-log", default=None)
    p_all.add_argument("--match-emails-file", default=None)
    p_all.add_argument("--match-emails-threshold", type=float, default=0.90)
    p_all.add_argument("--include-existing", action="store_true")
    p_all.add_argument("--ref-id", default=None)
    p_all.add_argument("--cache-ttl-hours", type=int, default=None)
    p_all.add_argument("--no-persist", action="store_true")
    p_all.add_argument("--include-checked", action="store_true")
    p_all.add_argument("--dry-run", action="store_true")
    p_all.set_defaults(func=run_all)

    p_sync_date = sub.add_parser("sync-from-date", help="Ad-hoc ingestion from a specific date")
    p_sync_date.add_argument("--start", required=True)
    p_sync_date.add_argument("--subject", default=None)
    p_sync_date.add_argument("--batch-size", type=int, default=1000)
    p_sync_date.add_argument("--limit", type=int, default=None)
    p_sync_date.add_argument("--max-seconds", type=int, default=None)
    p_sync_date.add_argument("--dry-run", action="store_true")
    p_sync_date.set_defaults(
        func=lambda args: sync_from_date_to_now(
            start_date=args.start,
            subject_text=args.subject,
            batch_size=args.batch_size,
            limit=args.limit,
            max_seconds=args.max_seconds,
            dry_run=args.dry_run,
        )
    )

    p_fetch = sub.add_parser("fetch-one", help="Fetch a single preprint by OSF id or DOI")
    g = p_fetch.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", help="OSF preprint id")
    g.add_argument("--doi", help="DOI or https://doi.org/... link")
    p_fetch.add_argument("--metadata-only", action="store_true", help="Only upsert metadata")
    p_fetch.add_argument("--skip-extract", action="store_true", help="Skip TEI extraction even after GROBID")
    p_fetch.set_defaults(
        func=lambda args: sync_one_by_id(
            osf_id=args.id,
            run_pdf_and_grobid=not args.metadata_only,
            run_extract=not args.skip_extract,
        )
        if args.id
        else sync_one_by_doi(
            doi_or_url=args.doi,
            run_pdf_and_grobid=not args.metadata_only,
            run_extract=not args.skip_extract,
        )
    )

    p_rand = sub.add_parser(
        "author-randomize",
        help="Assign unassigned preprints via a Dynamo-backed author network (initialize or augment)",
    )
    p_rand.add_argument(
        "--authors-csv",
        default="osf_sync/extraction/authorList_ext.csv",
        help="Optional enriched author CSV used for identity resolution (fallbacks to TEI/raw)",
    )
    p_rand.add_argument(
        "--limit-preprints",
        type=int,
        default=None,
        help="Optional cap on unassigned preprints processed in this run",
    )
    p_rand.add_argument("--seed", type=int, default=None, help="Explicit seed override")
    p_rand.add_argument(
        "--network-state-key",
        default="trial:author_network_state",
        help="sync_state key storing network metadata (x-threshold, next ids, seed chain)",
    )
    p_rand.add_argument("--dry-run", action="store_true")
    p_rand.set_defaults(
        func=lambda args: run_author_randomization(
            authors_csv=args.authors_csv,
            limit_preprints=args.limit_preprints,
            seed=args.seed,
            network_state_key=args.network_state_key,
            dry_run=args.dry_run,
        )
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "debug", False):
        logger.setLevel(logging.DEBUG)

    init_db()
    out = args.func(args)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
