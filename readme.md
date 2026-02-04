# OSF Preprints - Modular Pipeline

This repo runs a pipeline for OSF preprints. It does four big jobs:

1. Ingestion: fetch preprints from the OSF API and store them in DynamoDB (the database).
2. PDF + GROBID: download files and turn PDFs into TEI XML (GROBID is the PDF parser).
3. TEI extraction: parse TEI XML and write references back to DynamoDB.
4. Enrichment: look up missing DOIs using Crossref and OpenAlex.

If you are not sure what to run, follow "Quick Start" below.

## Quick Start (Local dev)

1. Fill `.env` (example below). For local dev, keep `DYNAMO_LOCAL_URL` and use fake AWS keys.
2. Start services:
   `docker compose up -d dynamodb-local redis`
   `docker compose up -d celery-worker celery-pdf celery-grobid celery-beat flower`
3. Create DynamoDB tables (run once):
   `docker compose run --rm app python -c "from osf_sync.db import init_db; init_db(); print('Dynamo tables ready')"`
4. Run a small sync:
   `docker compose run --rm app python -m osf_sync.cli sync-from-date --start 2025-07-01`
5. Queue the pipeline:
   `docker compose run --rm app python -m osf_sync.cli enqueue-pdf --limit 50`
   `docker compose run --rm app python -m osf_sync.cli enqueue-grobid --limit 25`
   `docker compose run --rm app python -m osf_sync.cli enqueue-extraction --limit 200`
6. Enrich references:
   `docker compose run --rm app python -m osf_sync.cli enrich-references --limit 400`

---

## Feature overview

- Incremental OSF ingestion with DynamoDB-backed cursors.
- PDF/GROBID pipeline with queue flags stored on each preprint.
- TEI parsing + reference extraction using the same data path.
- Enrichment via Crossref + OpenAlex using the multi-method pipeline.
- Extensive analytics scripts (no Docker required) to inspect coverage, dump references, or compute metrics.

---

## Requirements

- Docker Desktop 4GB+ RAM
- Python 3.11 (for local helpers) + pip/venv (optional)
- AWS CLI (optional, for querying AWS DynamoDB)
- OSF API token and optional AWS credentials (for hosted DynamoDB)

---

## Environment Configuration (`.env`)

```dotenv
GROBID_URL=http://grobid:8070
GROBID_INCLUDE_RAW_CITATIONS=true
CELERY_BROKER_URL=redis://redis:6379/0
AWS_REGION=eu-north-1
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
DDB_TABLE_PREPRINTS=preprints
DDB_TABLE_REFERENCES=preprint_references
DDB_TABLE_SYNCSTATE=sync_state
OPENALEX_EMAIL=<PERSONAL_EMAIL_ID>
PDF_DEST_ROOT=/data/preprints
LOG_LEVEL=INFO
```

- For **AWS DynamoDB**, comment out `DYNAMO_LOCAL_URL` and set real AWS credentials (or rely on an IAM role).
- The table names can be overridden with `DDB_TABLE_*` env vars.
- Set `GROBID_INCLUDE_RAW_CITATIONS=false` if you do not want raw citation sections in the TEI output.

---

## Running the ingestion stack

```bash
docker compose up -d dynamodb-local redis
docker compose up -d celery-worker celery-pdf celery-grobid celery-beat flower
```

- The `app` service is a helper container that initializes tables and prints CLI usage before exiting. Run one-off commands via `docker compose run --rm app ...`.
- Source code is bind-mounted into the containers, so edits in your IDE take effect immediately.

### Common commands

| Action                     | Command                                                                      |
| -------------------------- | ---------------------------------------------------------------------------- |
| Build images               | `docker compose build`                                                       |
| Start stack                | `docker compose up -d`                                                       |
| Inspect Celery worker logs | `docker compose logs -f celery-worker`                                       |
| Inspect PDF/GROBID logs    | `docker compose logs -f celery-pdf` / `docker compose logs -f celery-grobid` |
| Flower dashboard           | visit `http://localhost:5555`                                                |

---

## DynamoDB: Local vs AWS

### Local DynamoDB (default dev workflow)

1. `docker compose up -d dynamodb-local`
2. Ensure `.env` still contains `DYNAMO_LOCAL_URL=http://dynamodb-local:8000` and dummy AWS credentials.
3. Initialize tables/GSIs (only needed once):
   ```bash
   docker compose run --rm app python -c "from osf_sync.db import init_db; init_db(); print('Dynamo tables ready')"
   ```

### AWS-hosted DynamoDB

1. Comment/remove `DYNAMO_LOCAL_URL` in `.env`.
2. Export valid `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` (or rely on an IAM role).
3. Run `init_db()` once to create tables + GSIs in AWS:
   ```bash
   python -c "from osf_sync.db import init_db; init_db(); print('Dynamo tables ready')"
   ```
4. Restart services so they use the AWS endpoint.

> `init_db()` auto-creates tables and adds any missing GSIs. It is idempotent; rerun it after deploying schema changes.

---

## Inspecting DynamoDB Data

### Helper script (hosts & containers)

Run from repo root:

```bash
# Host (ensure env vars/venv installed)
python -m osf_sync.dump_ddb --limit 5 --queues

# Container
docker compose exec app python /app/osf_sync/dump_ddb.py --limit 5 --queues
```

Flags:

- `--table preprints` scans only that table.
- `--queues` queries `by_queue_pdf`, `by_queue_grobid`, `by_queue_extract` GSIs (falls back to scans if missing).

### AWS CLI snippets

Add `--endpoint-url http://localhost:8000` to point commands at local DynamoDB.

```powershell
# List tables
aws dynamodb list-tables --region eu-north-1

# Scan 10 preprints
aws dynamodb scan --table-name preprints --limit 10 --region eu-north-1

# Get a single preprint
aws dynamodb get-item --table-name preprints `
  --key '{"osf_id":{"S":"<OSF_ID>"}}' `
  --region eu-north-1

# Get a reference (composite key)
aws dynamodb get-item --table-name preprint_references `
  --key '{"osf_id":{"S":"<OSF_ID>"},"ref_id":{"S":"<REF_ID>"}}' `
  --region eu-north-1

# Queue queries
aws dynamodb query --table-name preprints `
  --index-name by_queue_pdf `
  --key-condition-expression "queue_pdf = :q" `
  --expression-attribute-values '{":q":{"S":"pending"}}' `
  --limit 20 --region eu-north-1

aws dynamodb query --table-name preprints `
  --index-name by_queue_grobid `
  --key-condition-expression "queue_grobid = :q" `
  --expression-attribute-values '{":q":{"S":"pending"}}' `
  --limit 20 --region eu-north-1

aws dynamodb query --table-name preprints `
  --index-name by_queue_extract `
  --key-condition-expression "queue_extract = :q" `
  --expression-attribute-values '{":q":{"S":"pending"}}' `
  --limit 20 --region eu-north-1
```

> The AWS PowerShell module (`AWS.Tools.DynamoDBv2`) exposes equivalent cmdlets (`Get-DDBItem`, `Get-DDBTableList`, etc.) if you prefer native PowerShell syntax.

---

## Task families

### Ingestion (Celery tasks/CLI)

### Sync OSF preprints

```bash
docker compose run --rm app python -m osf_sync.cli sync-from-date --start 2025-07-01
# Add --subject Psychology to filter
```

### Enqueue PDF downloads

```bash
docker compose run --rm app python -m osf_sync.cli enqueue-pdf --limit 50
docker compose logs -f celery-pdf
```

### PDF + GROBID + TEI (Celery tasks)

```bash
docker compose run --rm app python -m osf_sync.cli enqueue-grobid --limit 25
docker compose logs -f celery-grobid
```

### Parse existing TEI XML and write references

- After GROBID finishes, TEI files live under `/data/preprints/<provider>/<osf_id>/tei.xml`.
- `enqueue-extraction` walks that folder, parses each TEI XML, and writes structured TEI + reference records back into DynamoDB.

```bash
# Batch extraction (reads TEI XML directly, no OSF/API calls)
docker compose run --rm app python -m osf_sync.cli enqueue-extraction --limit 200

# Inspect logs
docker compose logs -f celery-worker
```

Each extraction job calls `osf_sync.augmentation.run_extract.extract_for_osf_id`, which:

1. Loads the TEI XML.
2. Uses the TEI parser to pull structured fields and references.
3. Persists output via `write_extraction`, updating `preprint_tei`, `preprint_references`, and `preprints.tei_extracted`.

### Enrichment tasks

```bash
docker compose run --rm app python -m osf_sync.cli enrich-references --limit 400
docker compose run --rm app python -m osf_sync.cli enrich-references --limit 400 --threshold 75
# Target a single reference:
# docker compose run --rm app python -m osf_sync.cli enrich-references --osf-id <OSF_ID> --ref-id <REF_ID> --debug
```

`celery-beat` already schedules the daily sync, PDF/GROBID queues, and enrichment tasks; adjust in `osf_sync/celery_app.py`.

---

## Analytics & local scripts (no Docker)

All scripts live under `scripts/manual_post_grobid/`. They work from the repo root after installing dependencies (`pip install -r requirements.txt`) and setting `PYTHONPATH`. Summary:

| Script                       | Description                                                              |
| ---------------------------- | ------------------------------------------------------------------------ |
| `run_extraction.py`          | Parse TEI XML from disk and write TEI/refs into DynamoDB.                |
| `doi_multi_method_lookup.py` | Run multi-method DOI matching and write a CSV. Does not update DynamoDB. |
| `run_forrt_screening.py`     | Run FORRT lookup + screening.                                            |
| `analyze_doi_sources.py`     | Count DOI coverage per source.                                           |
| `dump_missing_doi_refs.py`   | Dump references that still lack DOI information.                         |
| `select_low_doi_coverage.py` | Find OSF IDs below a DOI coverage threshold (optional ref dumps).        |

Run scripts with:

```bash
cd H:\fred_preprint_bot
$env:PYTHONPATH = "$PWD"    # PowerShell; use set/export on cmd/bash
python scripts/manual_post_grobid/analyze_doi_sources.py
python scripts/manual_post_grobid/dump_missing_doi_refs.py --output missing.jsonl
python scripts/manual_post_grobid/select_low_doi_coverage.py --threshold 0.2 --min-refs 30 --dump-refs-dir low_refs
```

See `scripts/manual_post_grobid/README.md` for details on each script.

---

## Manual scripts (no Docker required)

After configuring `.env` and installing dependencies locally, you can run the post-GROBID steps directly:

```bash
python scripts/manual_post_grobid/run_extraction.py --limit 200
python scripts/manual_post_grobid/doi_multi_method_lookup.py --from-db --limit 400 --output doi_multi_method.csv
python scripts/manual_post_grobid/run_forrt_screening.py --limit-lookup 200 --limit 500
python scripts/manual_post_grobid/analyze_doi_sources.py
python scripts/manual_post_grobid/dump_missing_doi_refs.py --output missing.jsonl
python scripts/manual_post_grobid/select_low_doi_coverage.py --threshold 0.2 --min-refs 30 --dump-refs-dir low_refs
```

Note: `doi_multi_method_lookup.py` writes a CSV only. To update DynamoDB, use the `enrich-references` task.

See `scripts/manual_post_grobid/README.md` for details and extra flags (`--dry-run`, `--sleep`, etc.).

---

## One-off Celery tasks

```bash
# Download a specific PDF
docker compose run --rm app sh -lc \
  "celery -A osf_sync.celery_app.app call osf_sync.tasks.download_single_pdf --args '[\"<OSF_ID>\"]'"

# Run GROBID for a specific preprint
docker compose run --rm app sh -lc \
  "celery -A osf_sync.celery_app.app call osf_sync.tasks.grobid_single --args '[\"<OSF_ID>\"]'"
```

---

## Quickstart Recap

```bash
# 1. Configure .env (local Dynamo or AWS)
# 2. Build + start services
docker compose build
docker compose up -d

# 3. Initialize DynamoDB tables/GSIs (first run)
docker compose run --rm app python -c "from osf_sync.db import init_db; init_db(); print('Dynamo tables ready')"

# 4. Sync, download PDFs, run GROBID
docker compose run --rm app python -m osf_sync.cli sync-from-date --start 2025-07-01
docker compose run --rm app python -m osf_sync.cli enqueue-pdf --limit 50
docker compose run --rm app python -m osf_sync.cli enqueue-grobid --limit 25

# 5. Parse TEI XML and enrich references
docker compose run --rm app python -m osf_sync.cli enqueue-extraction --limit 200
docker compose run --rm app python -m osf_sync.cli enrich-references --limit 400
```

Happy syncing!
