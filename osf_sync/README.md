# osf_sync package overview

Core pipeline package for OSF ingestion and enrichment without Celery.

## Key modules

| Module | Purpose |
| --- | --- |
| `pipeline.py` | Stage runner CLI and direct stage orchestration (`run`, `run-all`, `sync-from-date`, `fetch-one`). |
| `dynamo/` | Boto3 client, table definitions, and `PreprintsRepo` helpers for DynamoDB CRUD/state transitions. |
| `augmentation/` | TEI extraction and DOI/FLoRA enrichment utilities. |
| `fetch_one.py`, `iter_preprints.py` | OSF API helpers used by pipeline stages. |
| `pdf.py`, `grobid.py` | PDF download/conversion and TEI generation helpers. |

## CLI entrypoints

Use `python -m osf_sync.pipeline <command>`.

| Command | Description |
| --- | --- |
| `run --stage sync` | Incremental sync using the `sync_state` cursor. |
| `run --stage pdf` | Process `queue_pdf=pending` with claim/lease semantics. |
| `run --stage grobid` | Process `queue_grobid=pending` and generate TEI XML. |
| `run --stage extract` | Process `queue_extract=pending` and write TEI/references. |
| `run --stage enrich` | Multi-method DOI enrichment. |
| `run --stage flora` | FLoRA lookup + screening. |
| `run --stage author` | Author extraction and `author_email_candidates` updates. |
| `run-all` | Bounded sequential run across major stages, including `author` by default (`--skip-author` to disable). Author keeps files by default (`--cleanup-author-files` to delete) and is DynamoDB-only unless `--write-debug-csv` is set. |
| `sync-from-date` | Ad-hoc ingestion from a given start date. |
| `fetch-one` | Fetch one preprint by OSF id or DOI. |
| `author-randomize` | Assign only unassigned preprints using a Dynamo-backed author network (initializes on first run, augments thereafter). |

All commands support bounded execution (`--limit`, `--max-seconds`) and `--dry-run` where applicable.

## GitHub Actions orchestration

The pipeline runs automatically via `.github/workflows/pipeline.yml` on GH Actions hosted runners (`ubuntu-latest`).

**Schedule:** Daily at 03:00 UTC via cron, plus manual `workflow_dispatch`.

**Architecture:**
- GROBID runs as a GH Actions service container (`grobid/grobid:0.8.2.1-full` on port 8070)
- LibreOffice installed for DOCX-to-PDF conversion
- Filesystem is ephemeral; PDFs/TEI are re-downloaded or regenerated when missing on disk
- DynamoDB claim/lease system provides crash-safe resumption across runs
- Credentials are passed via GH Actions secrets as environment variables

**Parallelism** (separate worker counts per API destination):
- `--download-workers` (default 4): parallel PDF downloads from OSF
- `--enrich-workers` (default 6): parallel Crossref/OpenAlex DOI lookups
- `--orcid-workers` (default 3): parallel preprint processing for author/ORCID extraction
- GROBID is always sequential (single service container)

**Backlog handling:** After each run, if any stage was time-limited, the workflow re-triggers itself (up to `backlog_depth` times, default 5) to drain the queue. This requires a `WORKFLOW_DISPATCH_TOKEN` secret containing a PAT with `actions:write` scope, since `GITHUB_TOKEN` cannot trigger new workflow runs.

**Timeouts:** Job timeout is 330 minutes (within the 6-hour GH Actions limit). Per-stage limit defaults to 3600s for `run-all`.

## Queue/claim flow

`PreprintsRepo` queue status fields:
- `queue_pdf`
- `queue_grobid`
- `queue_extract`

Claim/lease fields used by queue stages:
- `claim_pdf_owner`, `claim_pdf_until`
- `claim_grobid_owner`, `claim_grobid_until`
- `claim_extract_owner`, `claim_extract_until`

Error/retry bookkeeping:
- `last_error_stage`, `last_error_message`, `last_error_at`
- `retry_count_pdf`, `retry_count_grobid`, `retry_count_extract`

## Ingestion filters

When `ingest.anchor_date` is set in `config/runtime.toml` (ISO date/timestamp), ingestion keeps only preprints whose `original_publication_date` (or fallback `date_published`) is within the configured window ending on the anchor date.

If a preprint has `links.doi` and that DOI is not OSF/Zenodo (`osf.io`, `zenodo.org`, or `10.5281/zenodo...`), it is skipped.

Set `OSF_INGEST_SKIP_EXISTING=true` to avoid upserting records already present in `preprints`.

## API cache table

`ensure_tables()` creates `api_cache` (or `DDB_TABLE_API_CACHE`) with TTL on `expires_at`.
Default TTL horizon is 6 months (`API_CACHE_TTL_MONTHS`).

## Exclusion counters

`ensure_tables()` also creates `excluded_preprints` (or `DDB_TABLE_EXCLUDED_PREPRINTS`), with one row per excluded `osf_id`.
Pipeline selectors skip preprints with `excluded=true`, and ingestion skips `osf_id`s present in `excluded_preprints`, so excluded preprints do not re-enter later stages.
