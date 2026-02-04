# osf_sync.dynamo

Plain English: this package wraps all DynamoDB reads and writes for the pipeline.

Utilities that encapsulate all DynamoDB access for the OSF pipeline.

---

## Files

| File             | Purpose                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------- |
| `client.py`      | Creates a boto3 DynamoDB resource. Switches between local (`DYNAMO_LOCAL_URL`) and AWS. |
| `tables.py`      | Declarative table + GSI specs and `ensure_tables()` helper (creates tables + GSIs).     |
| `preprints_repo.py` | High-level repository for CRUD, queue selection, and state transitions.                 |

---

## Client configuration (`client.py`)

```python
from osf_sync.dynamo.client import get_dynamo_resource

ddb = get_dynamo_resource()
table = ddb.Table("preprints")
```

- Local: set `DYNAMO_LOCAL_URL=http://dynamodb-local:8000` (or `http://localhost:8000` when running on host).
- AWS: omit `DYNAMO_LOCAL_URL` and ensure `AWS_REGION` + credentials (env vars or IAM role).
- All boto3 calls share a retry-aware `botocore.config.Config`.

---

## Table definitions (`tables.py`)

`TABLES` contains specs for:

- `preprints`: PK `osf_id`, GSIs `by_published`, `by_queue_pdf`, `by_queue_grobid`, `by_queue_extract`.
- `preprint_references`: composite PK (`osf_id`, `ref_id`), GSI `by_doi_source`.
- `preprint_tei`: PK `osf_id`.
- `sync_state`: PK `source_key`.

`ensure_tables()`:
1. Creates missing tables.
2. Calls `_ensure_gsis()` to add missing GSIs even on pre-existing tables (idempotent).

Run `init_db()` (from `osf_sync.db`) at startup to execute these steps automatically.

---

## Preprints repository (`preprints_repo.py`)

High-level API used across the codebase:

| Method                        | Description                                                                |
| ----------------------------- | -------------------------------------------------------------------------- |
| `upsert_preprints(rows)`      | Batch-writes OSF payloads, strips `None`, sets queue flags.                |
| `get_preprint_basic(osf_id)` | Returns `{osf_id, provider_id, raw}` projection.                           |
| `mark_pdf()` / `mark_tei()`   | Updates status flags + queue transitions (pdf->grobid->extract).            |
| `select_for_pdf(limit)`       | Query `by_queue_pdf` GSI (fallback to scan).                              |
| `select_for_grobid(limit)`    | Query `by_queue_grobid` GSI (fallback to scan).                           |
| `select_for_extraction(limit)`| Query `by_queue_extract` GSI (fallback to scan).                          |
| `set_cursor()/get_cursor()`   | Manages ISO timestamps in `sync_state`.                                   |
| `upsert_tei()` / `upsert_reference()` | Writes parsed TEI summaries and references.                 |
| `update_reference_doi()`      | Conditional DOI update (used by Crossref/OpenAlex enrichment).            |

The repo centralizes queue logic so that Celery tasks simply call `repo.select_for_*` to fetch work.

---

## Queue flow

1. OSF upsert sets `queue_pdf="pending"` for published preprints with a primary file.
2. `mark_pdf` sets `queue_pdf="done"` and `queue_grobid="pending"`.
3. `mark_tei` sets `queue_grobid="done"` and `queue_extract="pending"`.
4. `mark_extracted` sets `queue_extract="done"` after TEI parsing.

These attributes back the GSIs, giving efficient queue queries without scans.

---

## Testing / inspection

Use `python -m osf_sync.dump_ddb --limit 5 --queues` or AWS CLI commands to verify table contents and queue state.

