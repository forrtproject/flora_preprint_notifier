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

1. Create a virtual environment and install Python dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Install LibreOffice (`soffice`) locally if you need DOCX -> PDF conversion in the `pdf` stage.
3. Configure `.env`:
```bash
cp .env.example .env
```
4. Start local infrastructure services (optional if you use AWS DynamoDB and/or a remote GROBID):
```bash
docker compose up -d dynamodb-local grobid
```
5. Initialize DynamoDB tables:
```bash
python -c "from osf_sync.db import init_db; init_db(); print('Dynamo tables ready')"
```
6. Run pipeline stages:
```bash
python -m osf_sync.pipeline run --stage sync --limit 1000
python -m osf_sync.pipeline run --stage pdf --limit 100
python -m osf_sync.pipeline run --stage grobid --limit 50
python -m osf_sync.pipeline run --stage extract --limit 200
python -m osf_sync.pipeline run --stage enrich --limit 300
python -m osf_sync.pipeline run --stage flora --limit-lookup 200 --limit-screen 500
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
By default, `author` updates DynamoDB only (no local CSV output). Use `--write-debug-csv` (and optionally `--out`) for local debug snapshots.

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

Author-cluster randomisation (standalone, not in `run-all`):
```bash
python -m osf_sync.pipeline author-randomize \
  --network-state-key trial:author_network_state
```
Optionally add `--authors-csv <path>` to use an enriched author CSV if available.
Status: this workflow is not yet validated end-to-end in production and should be treated as experimental.
This command processes only unassigned preprints.
If no prior trial network exists, it initializes one from those preprints; otherwise it loads the latest network from DynamoDB and augments it.
Allocations, graph state, and run metadata are stored in DynamoDB trial tables plus `sync_state`.
Use `--dry-run` to preview candidate processing and allocation counts without writing to DynamoDB:
```bash
python -m osf_sync.pipeline author-randomize --dry-run
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
- `--write-debug-csv` (`author` stage): write a local debug CSV snapshot (`--out` overrides the default path).

## Environment (`.env`)

```dotenv
# local Docker GROBID:
GROBID_URL=http://localhost:8070
# remote GROBID example:
# GROBID_URL=https://grobid.example.org
GROBID_INCLUDE_RAW_CITATIONS=true
DYNAMO_LOCAL_URL=http://localhost:8000
AWS_REGION=eu-north-1
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
DDB_TABLE_PREPRINTS=preprints
DDB_TABLE_REFERENCES=preprint_references
DDB_TABLE_EXCLUDED_PREPRINTS=excluded_preprints
DDB_TABLE_SYNCSTATE=sync_state
DDB_TABLE_API_CACHE=api_cache
OPENALEX_EMAIL=<PERSONAL_EMAIL_ID>
PDF_DEST_ROOT=./data/preprints
LOG_LEVEL=INFO
OSF_INGEST_ANCHOR_DATE=YYYY-MM-DD
OSF_INGEST_SKIP_EXISTING=false
API_CACHE_TTL_MONTHS=6
FLORA_CSV_PATH=./data/flora.csv
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

## DOI Experiment Command

Use the module entrypoint directly for DOI matching experiments:

```bash
python -m osf_sync.augmentation.doi_multi_method_lookup --from-db --limit 400 --output doi_multi_method.csv
```
