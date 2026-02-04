# osf_sync package overview

This is the main Python package. It contains the background jobs (Celery tasks), the command line entrypoints, and the DynamoDB helpers used by the pipeline.

---

## Key modules

| Module                              | Purpose                                                                           |
| ----------------------------------- | --------------------------------------------------------------------------------- |
| `celery_app.py`                     | Celery configuration + beat schedules (+ queue routing).                          |
| `cli.py`                            | Argparse-based CLI invoked via `python -m osf_sync.cli ...`.                      |
| `tasks.py`                          | Celery tasks (OSF sync, PDF download, GROBID, TEI extraction, enrichments, etc.). |
| `dynamo/`                           | Boto3 client, table definitions, and `PreprintsRepo` for DynamoDB CRUD helpers.   |
| `augmentation/`                     | TEI parser integration and enrichment utilities (Crossref, OpenAlex, etc.).       |
| `dump_ddb.py`                       | Helper script to scan/query DynamoDB tables & GSIs.                               |
| `fetch_one.py`, `iter_preprints.py` | OSF API HTTP helpers used by CLI/tasks.                                           |
| `pdf.py`, `grobid.py`               | PDF download + TEI generation helpers.                                            |

---

## Ingestion filters

When `OSF_INGEST_ANCHOR_DATE` is set (ISO date or timestamp), ingestion only keeps
preprints whose `original_publication_date` (if present) or `date_published` falls
within the 6-month window ending on the anchor date. If a preprint has `links.doi`
and that DOI link is not an OSF or Zenodo link (`osf.io`, `zenodo.org`, or a
Zenodo DOI like `10.5281/zenodo...`), it is skipped.

Set `OSF_INGEST_SKIP_EXISTING=true` to skip upserting records that already exist
in the preprints table (adds a read-before-write check).

---

## API cache table

`ensure_tables()` now creates an `api_cache` table (or `DDB_TABLE_API_CACHE` override)
with PK `cache_key` and enables TTL on the `expires_at` attribute. Default TTL
horizon is 6 months via `API_CACHE_TTL_MONTHS` (see `osf_sync/dynamo/api_cache_repo.py`).

---

## CLI entrypoints

Run everything via `python -m osf_sync.cli <command> [options]`.

| Command                                          | Description                                                  |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `sync-from-date`                                 | Incremental OSF sync from a start date (optional subject).   |
| `enqueue-pdf`                                    | Queue PDFs for download.                                     |
| `enqueue-grobid`                                 | Queue GROBID jobs for downloaded PDFs.                       |
| `enqueue-extraction`                             | Parse TEI XML from disk and write TEI/refs to DynamoDB.      |
| `enrich-references`                              | Fill missing reference DOIs using the multi-method pipeline. |
| `enrich-references --osf-id <ID> --ref-id <RID>` | Re-run enrichment for a single reference.                    |
| `fetch-one`                                      | Fetch a single OSF preprint by ID/DOI and upsert it.         |

Each CLI command wraps Celery tasks or helper functions defined inside this package.

---

## Tasks & queues

| Task name                              | Queue    | Notes                                                    |
| -------------------------------------- | -------- | -------------------------------------------------------- |
| `osf_sync.tasks.sync_from_osf`         | default  | OSF incremental sync, uses DynamoDB `sync_state` cursor. |
| `osf_sync.tasks.enqueue_pdf_downloads` | default  | Selects items via `preprints.by_queue_pdf` GSI.          |
| `osf_sync.tasks.download_single_pdf`   | `pdf`    | Downloads+converts files, updates Dynamo flags.          |
| `osf_sync.tasks.enqueue_grobid`        | default  | Selects items via `preprints.by_queue_grobid` GSI.       |
| `osf_sync.tasks.grobid_single`         | `grobid` | Runs GROBID and marks `queue_grobid`/`queue_extract`.    |
| `osf_sync.tasks.enqueue_extraction`    | default  | Selects items via `preprints.by_queue_extract` GSI.      |
| `osf_sync.tasks.enrich_references`     | default  | Updates references via the multi-method DOI pipeline.    |

The `PreprintsRepo` class centralizes all DynamoDB CRUD and queue-selection logic so tasks remain concise.

---

## Augmentation pipeline

1. `grobid.py` saves TEI XML to `/data/preprints/<provider>/<osf_id>/tei.xml`.
2. `augmentation/run_extract.py` parses each TEI file with the repo's TEI extractor.
3. `augmentation/extract_to_db.py` writes structured TEI and reference data through `PreprintsRepo`.
4. Enrichment uses the multi-method DOI pipeline to fill missing DOIs.

All augmentation modules assume DynamoDB as the backing store and use the helpers in `dynamo/preprints_repo.py`.

---

## Helper scripts

- `dump_ddb.py` - scan/query Dynamo tables and GSIs for quick inspection (works in containers and on host).
- `augmentation/run_extract.py` - entrypoint for parsing a single TEI file (used by Celery tasks).
- Scripts under `scripts/manual_post_grobid/` are for running steps by hand (no Docker).

Use `python -m osf_sync.dump_ddb --help` for inspection options.
