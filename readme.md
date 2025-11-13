# OSF Preprints → DynamoDB → PDF/GROBID

End-to-end pipeline that ingests OSF preprints, stores metadata in **DynamoDB**, downloads primary PDFs, and extracts TEI XML via **GROBID** – orchestrated with **Celery** and **Docker Compose**.

---

## Features

- Incremental OSF sync with per-provider cursors stored in DynamoDB
- PDF download + DOCX→PDF conversion with per-preprint state flags
- GROBID processing and TEI reference extraction
- Queue selection backed by DynamoDB GSIs (no table scans)
- Works with **local DynamoDB** (default) or **AWS-hosted DynamoDB** by toggling env vars

---

## Requirements

- Docker Desktop 4GB+ RAM
- Python 3.11 (for local helpers) + pip/venv (optional)
- AWS CLI (optional, for querying AWS DynamoDB)
- OSF API token and optional AWS credentials (for hosted DynamoDB)

---

## Environment Configuration (`.env`)

```dotenv
# Core services
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
PDF_DEST_ROOT=/data/preprints
GROBID_URL=http://grobid:8070
LOG_LEVEL=INFO

# OSF + enrichment
OSF_API_TOKEN=...                # required for OSF sync
OPENALEX_EMAIL=you@example.com

# DynamoDB (local dev defaults)
AWS_REGION=eu-north-1
AWS_ACCESS_KEY_ID=local
AWS_SECRET_ACCESS_KEY=local
DYNAMO_LOCAL_URL=http://dynamodb-local:8000   # comment/remove when using AWS-hosted DynamoDB
DDB_TABLE_PREPRINTS=preprints
DDB_TABLE_REFERENCES=preprint_references
DDB_TABLE_TEI=preprint_tei
DDB_TABLE_SYNCSTATE=sync_state
```

- For **AWS DynamoDB**, comment out `DYNAMO_LOCAL_URL` and set real AWS credentials (or rely on an IAM role).
- The table names can be overridden with `DDB_TABLE_*` env vars.

---

## Running the Stack

```bash
docker compose up -d dynamodb-local redis
docker compose up -d celery-worker celery-pdf celery-grobid celery-beat flower
```

- The `app` service is a helper container that initializes tables and prints CLI usage before exiting. Run one-off commands via `docker compose run --rm app ...`.
- Source code is bind-mounted into the containers, so edits in your IDE take effect immediately.

### Common commands

| Action                        | Command                                                                 |
| ----------------------------- | ----------------------------------------------------------------------- |
| Build images                  | `docker compose build`                                                  |
| Start stack                   | `docker compose up -d`                                                  |
| Inspect Celery worker logs    | `docker compose logs -f celery-worker`                                  |
| Inspect PDF/GROBID logs       | `docker compose logs -f celery-pdf` / `docker compose logs -f celery-grobid` |
| Flower dashboard              | visit `http://localhost:5555`                                           |

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

## Core Workflows

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

### Enqueue GROBID processing

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
docker compose run --rm app python -m osf_sync.cli enrich-crossref --limit 400
docker compose run --rm app python -m osf_sync.cli enrich-openalex --limit 400 --threshold 75
```

`celery-beat` already schedules the daily sync, PDF/GROBID queues, and enrichment tasks; adjust in `osf_sync/celery_app.py`.

---

## Troubleshooting Tips

| Symptom                                   | Fix                                                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `service "app" is not running`            | Expected – `app` runs init + CLI help then exits. Use `docker compose run --rm app ...` as needed.  |
| `ValidationException` querying GSIs       | Ensure `init_db()` was run so GSIs exist, or delete local data dir (`rm -rf .dynamodb`) and restart |
| `Unable to locate credentials`            | Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, and (for local) `DYNAMO_LOCAL_URL`. |
| Local edits not reflected in containers   | Source is bind-mounted; restart the service if Python module caching causes issues.                 |
| Need to inspect Dynamo data quickly       | `python -m osf_sync.dump_ddb --limit 5 --queues` (host) or `docker compose exec app ...`            |

---

## One-off Celery Tasks

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
docker compose run --rm app python -m osf_sync.cli enrich-crossref --limit 400
docker compose run --rm app python -m osf_sync.cli enrich-openalex --limit 400 --threshold 75
```

Happy syncing!
