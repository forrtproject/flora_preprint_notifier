# osf_sync package overview

Core Python package for the OSF Preprints pipeline. It contains the Celery tasks, CLI entrypoints, DynamoDB helpers, and augmentation utilities used by the stack.

---

## Key modules

| Module                          | Purpose                                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------------------- |
| `celery_app.py`                 | Celery configuration + beat schedules (+ queue routing).                                 |
| `cli.py`                        | Click-based CLI invoked via `python -m osf_sync.cli …`.                                   |
| `tasks.py`                      | Celery tasks (OSF sync, PDF download, GROBID, TEI extraction, enrichments, etc.).        |
| `dynamo/`                       | Boto3 client, table definitions, and `PreprintsRepo` for DynamoDB CRUD helpers.         |
| `augmentation/`                 | TEI parser integration and enrichment utilities (Crossref, OpenAlex, etc.).             |
| `dump_ddb.py`                   | Helper script to scan/query DynamoDB tables & GSIs.                                      |
| `fetch_one.py`, `iter_preprints.py` | OSF API HTTP helpers used by CLI/tasks.                                                 |
| `pdf.py`, `grobid.py`           | PDF download + TEI generation helpers.                                                   |

---

## CLI entrypoints

Run everything via `python -m osf_sync.cli <command> [options]`.

| Command               | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `sync-from-date`      | Incremental OSF sync from a start date (optional subject). |
| `enqueue-pdf`         | Queue PDFs for download.                               |
| `enqueue-grobid`      | Queue GROBID jobs for downloaded PDFs.                 |
| `enqueue-extraction`  | Parse TEI XML from disk and write TEI/refs to DynamoDB.|
| `enrich-crossref`     | Fill missing reference DOIs via Crossref.              |
| `enrich-openalex`     | Remaining DOI enrichment via OpenAlex.                |
| `enrich-crossref --osf-id <ID> --ref-id <RID>` | Re-run Crossref enrichment for a single reference. |
| `fetch-one`           | Fetch a single OSF preprint by ID/DOI and upsert it.   |

Each CLI command wraps Celery tasks or helper functions defined inside this package.

---

## Tasks & queues

| Task name                                 | Queue    | Notes                                                           |
| ----------------------------------------- | -------- | ----------------------------------------------------------------|
| `osf_sync.tasks.sync_from_osf`            | default  | OSF incremental sync, uses DynamoDB `sync_state` cursor.        |
| `osf_sync.tasks.enqueue_pdf_downloads`    | default  | Selects items via `preprints.by_queue_pdf` GSI.                 |
| `osf_sync.tasks.download_single_pdf`      | `pdf`    | Downloads+converts files, updates Dynamo flags.                 |
| `osf_sync.tasks.enqueue_grobid`           | default  | Selects items via `preprints.by_queue_grobid` GSI.              |
| `osf_sync.tasks.grobid_single`            | `grobid` | Runs GROBID and marks `queue_grobid`/`queue_extract`.           |
| `osf_sync.tasks.enqueue_extraction`       | default  | Selects items via `preprints.by_queue_extract` GSI.             |
| `osf_sync.tasks.enrich_crossref`          | default  | Updates `preprint_references` via `matching_crossref`.         |
| `osf_sync.tasks.enrich_openalex`          | default  | Updates references via OpenAlex API, custom threshold/mailto.  |

The `PreprintsRepo` class centralizes all DynamoDB CRUD and queue-selection logic so tasks remain concise.

---

## Augmentation pipeline

1. `grobid.py` saves TEI XML to `/data/preprints/<provider>/<osf_id>/tei.xml`.
2. `augmentation/run_extract.py` parses each TEI file with the repo’s TEI extractor.
3. `augmentation/extract_to_db.py` writes structured TEI and reference data through `PreprintsRepo`.
4. Enrichment tasks (`matching_crossref`, `doi_check_openalex`, `enrich_doi`) fill missing DOIs.

All augmentation modules assume DynamoDB as the backing store and use the helpers in `dynamo/preprints_repo.py`.

---

## Helper scripts

- `dump_ddb.py` – scan/query Dynamo tables and GSIs for quick inspection (works in containers and on host).
- `augmentation/run_extract.py` – entrypoint for parsing a single TEI file (used by Celery tasks).
- `augmentation/*` scripts (under `scripts/augmentation/`) – legacy helpers usable from the container if needed.

Use `python -m osf_sync.dump_ddb --help` for inspection options.

