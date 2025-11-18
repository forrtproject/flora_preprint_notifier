# osf_sync/tasks.py
from __future__ import annotations

import os
import datetime as dt
import json
import logging
import time
from typing import Optional, Dict, Any, Iterable

import requests
from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import RequestException
from .dynamo.preprints_repo import PreprintsRepo

from celery import chain

from .celery_app import app
from .db import init_db
from .upsert import upsert_batch
from .iter_preprints import iter_preprints_batches, iter_preprints_range

# Your existing helper modules
from .pdf import mark_downloaded, ensure_pdf_available_or_delete
from .grobid import process_pdf_to_tei, mark_tei

from .fetch_one import fetch_preprint_by_id, fetch_preprint_by_doi, upsert_one_preprint

# -------------------------------
# Config / env
# -------------------------------
OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", "you@example.com")
PDF_DEST_ROOT = os.environ.get("PDF_DEST_ROOT", "/data/preprints")
OSF_API = os.environ.get("OSF_API", "https://api.osf.io/v2")
SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL")  # optional

SOURCE_KEY_ALL = "osf:all"

# HTTP config
HTTP_RETRIES = Retry(
    total=5, backoff_factor=0.6,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET", "POST"),
)
HTTP_TIMEOUT = 60

# -------------------------------
# Logging
# -------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logger.setLevel(logging.INFO)


def _make_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=HTTP_RETRIES))
    s.mount("http://", HTTPAdapter(max_retries=HTTP_RETRIES))
    token = os.environ.get("OSF_API_TOKEN")
    if token:
        s.headers.update({"Authorization": f"Bearer {token}"})
    s.headers.update({"User-Agent": "osf_sync/1.2"})
    return s


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


# -------------------------------
# Helpers: cursor state in Postgres
# -------------------------------
def _coerce_osf_id(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("osf_id") or x.get("id")
    raise ValueError(f"Unsupported argument for osf_id: {type(x)}")


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


# -------------------------------
# Public tasks
# -------------------------------
@app.task
def init_schema() -> str:
    init_db()
    return "OK"


@app.task(
    bind=True,
    autoretry_for=(RequestException,),
    retry_backoff=30,
    retry_jitter=True,
    retry_kwargs={"max_retries": 5},
)
def sync_from_osf(
    self,
    subject_text: Optional[str] = None,
    batch_size: int = 1000,
) -> Dict[str, str | int | None]:
    """
    Daily incremental sync based on date_published cursor per subject.
    """
    init_db()
    source_key = f"osf:{subject_text}" if subject_text else SOURCE_KEY_ALL

    since_dt = _get_cursor(source_key)
    if since_dt is None:
        since_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)

    since_iso_date = since_dt.astimezone(dt.timezone.utc).date().isoformat()

    total_upserted = 0
    max_published_seen: Optional[dt.datetime] = None

    logger.info("sync_from_osf start", extra={"since": since_iso_date, "subject": subject_text})
    for batch in iter_preprints_batches(
        since_date=since_iso_date,
        subject_text=subject_text,
        batch_size=batch_size,
        sort="date_published",
    ):
        for obj in batch:
            pub = _parse_iso_dt((obj.get("attributes") or {}).get("date_published"))
            if pub and (max_published_seen is None or pub > max_published_seen):
                max_published_seen = pub

        total_upserted += upsert_batch(batch)
        logger.info("upserted batch", extra={"batch_size": len(batch), "total": total_upserted})
        time.sleep(0.2)

    cursor_out = (max_published_seen or since_dt).isoformat()
    if max_published_seen:
        _set_cursor(source_key, max_published_seen)

    out = {"upserted": total_upserted, "cursor": cursor_out}
    _slack("OSF sync finished", extra=out)
    return out


@app.task(bind=True)
def sync_from_date_to_now(
    self, start_date: str, subject_text: str | None = None, batch_size: int = 1000
):
    """
    Ad-hoc ingestion from start_date (YYYY-MM-DD) to today. Does not touch sync_state.
    """
    init_db()
    total = 0
    logger.info("sync_from_date_to_now", extra={"start": start_date, "subject": subject_text})
    for batch in iter_preprints_range(
        start_date=start_date, until_date=None, subject_text=subject_text, batch_size=batch_size
    ):
        total += upsert_batch(batch)
        logger.info("upserted batch", extra={"batch_size": len(batch), "total": total})
    out = {"upserted": total, "from": start_date, "to": "now"}
    _slack("Ad-hoc OSF window sync finished", extra=out)
    return out


# -------------------------------
# PDF download queue + worker
# -------------------------------
@app.task(bind=True, autoretry_for=(RequestException,), retry_backoff=30, retry_jitter=True, retry_kwargs={"max_retries": 3})
def download_single_pdf(self, osf_id: str):
    """
    Only PDF/DOCX allowed (DOCX converted to PDF inside ensure_pdf_available_or_delete).
    Others: delete row or mark excluded (handled inside your .pdf module).
    """
    repo = PreprintsRepo()
    row = repo.get_preprint_basic(osf_id)
    if not row:
        return {"osf_id": osf_id, "skipped": "no longer in DB"}

    provider_id = row["provider_id"] or "unknown"

    # Your helper should now resolve relationships.primary_file → file JSON → data.links.download
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


@app.task(bind=True)
def enqueue_pdf_downloads(self, limit: int = 100):
    repo = PreprintsRepo()
    ids = repo.select_for_pdf(limit)

    if not ids:
        return {"queued": 0}

    # Execute strictly one-by-one (no overlap) using a chain of immutable signatures
    sigs = [download_single_pdf.si(i) for i in ids]
    chain(*sigs).apply_async(queue="pdf")
    out = {"queued": len(ids)}
    _slack("Enqueued PDF downloads", extra=out)
    return out


# -------------------------------
# GROBID queue + worker
# -------------------------------
@app.task(bind=True, autoretry_for=(RequestException,), retry_backoff=30, retry_jitter=True, retry_kwargs={"max_retries": 3})
def grobid_single(self, osf_id_or_result):
    """
    Accepts either a plain osf_id or the previous task's dict result containing 'osf_id'.
    """
    osf_id = _coerce_osf_id(osf_id_or_result)
    if not osf_id:
        return {"ok": False, "error": "missing osf_id in previous result"}

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
    mark_tei(osf_id, ok=ok, tei_path=tei_path if ok else None)
    logger.info("GROBID done", extra={"osf_id": osf_id, "ok": ok, "tei": tei_path})
    return {"osf_id": osf_id, "ok": ok, "tei_path": tei_path, "error": err}


@app.task(bind=True)
def enqueue_grobid(self, limit: int = 50):
    """
    Queue a strictly sequential chain of GROBID jobs.
    """
    repo = PreprintsRepo()
    ids = repo.select_for_grobid(limit)
    if not ids:
        return {"queued": 0}

    sigs = [grobid_single.si(i) for i in ids]
    chain(*sigs).apply_async(queue="grobid")
    out = {"queued": len(ids)}
    _slack("Enqueued GROBID jobs", extra=out)
    return out


# -------------------------------
# One-shot helpers (by id/doi)
# -------------------------------
@app.task(name="osf_sync.tasks.sync_one_by_id", bind=True)
def sync_one_by_id(self, osf_id: str, run_pdf_and_grobid: bool = True):
    data = fetch_preprint_by_id(osf_id)
    if not data:
        return {"ok": False, "reason": "not found", "osf_id": osf_id}
    upserted = upsert_one_preprint(data)
    result = {"ok": True, "osf_id": data["id"], "upserted": upserted}

    if run_pdf_and_grobid:
        ch = chain(
            download_single_pdf.s(data["id"]),
            grobid_single.s(),
            extract_from_tei.s(),  # this one now accepts either (provider_id, osf_id) or a prior dict
        )
        async_res = ch.apply_async()
        result["chain_id"] = async_res.id
    return result


@app.task(name="osf_sync.tasks.sync_one_by_doi", bind=True)
def sync_one_by_doi(self, doi_or_url: str, run_pdf_and_grobid: bool = True):
    data = fetch_preprint_by_doi(doi_or_url)
    if not data:
        return {"ok": False, "reason": "not found", "doi": doi_or_url}
    upserted = upsert_one_preprint(data)
    result = {"ok": True, "osf_id": data["id"], "upserted": upserted}

    if run_pdf_and_grobid:
        ch = chain(
            download_single_pdf.s(data["id"]),
            grobid_single.s(),
            extract_from_tei.s(),
        )
        async_res = ch.apply_async()
        result["chain_id"] = async_res.id
    return result


# -------------------------------
# TEI extraction
# -------------------------------
@app.task(name="osf_sync.tasks.extract_from_tei", bind=True, queue="grobid")
def extract_from_tei(self, maybe_provider_or_result, maybe_osf_id: Optional[str] = None):
    """
    Accepts:
      - (provider_id, osf_id) OR
      - a dict result from previous task containing 'osf_id' (we'll look up provider_id)
    """
    from .augmentation.run_extract import extract_for_osf_id

    if isinstance(maybe_provider_or_result, dict):
        osf_id = maybe_provider_or_result.get("osf_id")
        if not osf_id:
            return {"ok": False, "error": "missing osf_id in prior result"}
        repo = PreprintsRepo()
        b = repo.get_preprint_basic(osf_id)
        if not b:
            return {"ok": False, "error": "osf_id not found"}
        provider_id = b.get("provider_id")
    else:
        provider_id = maybe_provider_or_result
        osf_id = maybe_osf_id

    base = os.environ.get("PDF_DEST_ROOT", PDF_DEST_ROOT)
    n = extract_for_osf_id(provider_id, osf_id, base)
    logger.info("TEI extracted", extra={"osf_id": osf_id, "refs": n})
    return {"osf_id": osf_id, "references_upserted": n}


@app.task(name="osf_sync.tasks.enqueue_extraction", bind=True)
def enqueue_extraction(self, limit: int = 200):
    repo = PreprintsRepo()
    items = repo.select_for_extraction(limit)
    for it in items:
        extract_from_tei.apply_async(args=[it.get("provider_id"), it.get("osf_id")], queue="grobid")
    out = {"queued": len(items)}
    _slack("Enqueued TEI extraction", extra=out)
    return out


# -------------------------------
# Enrichment (Crossref ↔ OpenAlex)
# -------------------------------
@app.task(name="osf_sync.tasks.enrich_crossref", bind=True)
def enrich_crossref(
    self,
    limit: int = 300,
    ua_email: str = OPENALEX_EMAIL,
    osf_id: str | None = None,
    ref_id: str | None = None,
    debug: bool = False,
):
    from .augmentation.matching_crossref import enrich_missing_with_crossref
    stats = enrich_missing_with_crossref(
        limit=limit,
        ua_email=ua_email,
        osf_id=osf_id,
        ref_id=ref_id,
        debug=debug,
    )
    _slack("Crossref enrichment", extra=stats)
    return stats


@app.task(name="osf_sync.tasks.enrich_openalex", bind=True)
def enrich_openalex(
    self,
    limit: int = 200,
    threshold: int = 70,
    mailto: Optional[str] = None,
    osf_id: Optional[str] = None,
    debug: bool = False,
):
    from .augmentation.doi_check_openalex import enrich_missing_with_openalex

    stats = enrich_missing_with_openalex(
        limit=limit,
        threshold=threshold,
        mailto=mailto,
        osf_id=osf_id,
        debug=debug,
    )
    _slack("OpenAlex enrichment", extra=stats)
    return stats
