# osf_sync.dynamo

DynamoDB access layer for the OSF pipeline.

## Files

| File | Purpose |
| --- | --- |
| `client.py` | Creates a boto3 DynamoDB resource (local or AWS). |
| `tables.py` | Table + GSI specs and `ensure_tables()` helper. |
| `preprints_repo.py` | High-level repository for queue selection, claims, and state transitions. |
| `api_cache_repo.py` | TTL-backed cache table helper. |

## Tables

`ensure_tables()` manages:
- `preprints` (GSIs: `by_published`, `by_queue_pdf`, `by_queue_grobid`, `by_queue_extract`)
- `preprint_references`
- `preprint_tei`
- `sync_state`
- `api_cache` (TTL on `expires_at`)

## PreprintsRepo highlights

- `upsert_preprints(rows)`
- `select_for_pdf(limit)` / `select_for_grobid(limit)` / `select_for_extraction(limit)`
- `mark_pdf()` / `mark_tei()` / `mark_extracted()`
- `claim_stage_item(stage, osf_id, owner, lease_seconds)`
- `release_stage_claim(stage, osf_id)`
- `record_stage_error(stage, osf_id, message)`

Queue flow:
1. `queue_pdf=pending`
2. `queue_grobid=pending`
3. `queue_extract=pending`
4. done

Claim fields (`claim_*_owner`, `claim_*_until`) support safe concurrent schedulers.

## Example usage

```python
from osf_sync.db import init_db
from osf_sync.dynamo.preprints_repo import PreprintsRepo

init_db()
repo = PreprintsRepo()
print(repo.select_for_pdf(limit=10))
```
