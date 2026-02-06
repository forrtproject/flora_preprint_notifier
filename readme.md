# OSF Preprints - Modular Pipeline (No Celery)

This repository runs a bounded, stage-based pipeline for OSF preprints using DynamoDB as the single source of truth.

Pipeline stages:
1. `sync`: ingest preprints from OSF
2. `pdf`: download/convert primary files
3. `grobid`: generate TEI from PDFs
4. `extract`: parse TEI and write references
5. `enrich`: fill missing reference DOIs
6. `flora`: FLoRA lookup + screening
7. `author`: author/email candidate extraction

All stages run as normal Python commands and exit. Scheduling is external (cron or GitHub Actions).
The `flora` stage checks whether originals have replications cited in the FLoRA database (the FORRT Library of Replication Attempts).

## Quick Start (Local)

1. Configure `.env`.
2. Start local dependencies:
```bash
docker compose up -d dynamodb-local grobid
```
3. Initialize DynamoDB tables:
```bash
docker compose run --rm app python -c "from osf_sync.db import init_db; init_db(); print('Dynamo tables ready')"
```
4. Run pipeline stages:
```bash
docker compose run --rm app python -m osf_sync.pipeline run --stage sync --limit 1000
docker compose run --rm app python -m osf_sync.pipeline run --stage pdf --limit 100
docker compose run --rm app python -m osf_sync.pipeline run --stage grobid --limit 50
docker compose run --rm app python -m osf_sync.pipeline run --stage extract --limit 200
docker compose run --rm app python -m osf_sync.pipeline run --stage enrich --limit 300
docker compose run --rm app python -m osf_sync.pipeline run --stage flora --limit-lookup 200 --limit-screen 500
```

## Main Commands

Single stage:
```bash
python -m osf_sync.pipeline run --stage <sync|pdf|grobid|extract|enrich|flora|author> [options]
```

Full bounded run:
```bash
python -m osf_sync.pipeline run-all \
  --sync-limit 1000 --pdf-limit 100 --grobid-limit 50 --extract-limit 200 --enrich-limit 300
```
`run-all` includes the `author` stage by default; use `--skip-author` to disable it for a run.
By default, `run-all` keeps local PDF/TEI files during `author`; use `--cleanup-author-files` to allow cleanup.

Ad-hoc window sync:
```bash
python -m osf_sync.pipeline sync-from-date --start 2025-07-01
```

One-off preprint:
```bash
python -m osf_sync.pipeline fetch-one --id <OSF_ID>
# or
python -m osf_sync.pipeline fetch-one --doi <DOI_OR_URL>
```

`python -m osf_sync.cli ...` is now a thin alias to the same pipeline CLI.

## Common Options

- `--limit`: max items for the stage.
- `--max-seconds`: stop the stage after N seconds.
- `--dry-run`: compute/select work without executing mutations.
- `--debug`: enable verbose logging.
- `--owner` and `--lease-seconds` (queue stages): override DynamoDB claim ownership/lease duration.
- `--skip-author` (`run-all`): skip author extraction when needed.
- `--cleanup-author-files` (`run-all`): allow author stage file deletion (off by default).

## Environment (`.env`)

```dotenv
GROBID_URL=http://grobid:8070
GROBID_INCLUDE_RAW_CITATIONS=true
AWS_REGION=eu-north-1
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
DDB_TABLE_PREPRINTS=preprints
DDB_TABLE_REFERENCES=preprint_references
DDB_TABLE_SYNCSTATE=sync_state
DDB_TABLE_API_CACHE=api_cache
OPENALEX_EMAIL=<PERSONAL_EMAIL_ID>
PDF_DEST_ROOT=/data/preprints
LOG_LEVEL=INFO
OSF_INGEST_ANCHOR_DATE=YYYY-MM-DD
OSF_INGEST_SKIP_EXISTING=false
API_CACHE_TTL_MONTHS=6
PIPELINE_CLAIM_LEASE_SECONDS=1800
```

## Scheduling

Use either:
- Cron/systemd timers on a VM, or
- GitHub Actions `schedule` workflows.

Recommended pattern:
- Run each stage independently on a cadence with bounded limits.
- Allow overlap; claim/lease fields in DynamoDB prevent duplicate processing.

## DynamoDB Queue Flow

1. `sync` sets `queue_pdf=pending` when eligible.
2. `pdf` marks `queue_pdf=done`, `queue_grobid=pending`.
3. `grobid` marks `queue_grobid=done`, `queue_extract=pending`.
4. `extract` marks `queue_extract=done`.

Queue stages use claim/lease metadata (`claim_*_owner`, `claim_*_until`) and error tracking fields (`last_error_*`, `retry_count_*`).

## Manual Post-GROBID Scripts

Scripts under `scripts/manual_post_grobid/` still work for ad-hoc analysis and downstream jobs.

Examples:
```bash
python scripts/manual_post_grobid/run_extraction.py --limit 200
python scripts/manual_post_grobid/doi_multi_method_lookup.py --from-db --limit 400 --output doi_multi_method.csv
python scripts/manual_post_grobid/run_flora_screening.py --limit-lookup 200 --limit 500
python scripts/manual_post_grobid/enqueue_author_extract.py --limit 100
```
