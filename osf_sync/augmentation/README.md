# osf_sync.augmentation

Utilities that parse TEI XML, write structured data back to DynamoDB, and enrich references via Crossref / OpenAlex.

---

## TEI parsing & extraction

| File                  | Purpose                                                                                       |
| --------------------- | --------------------------------------------------------------------------------------------- |
| `run_extract.py`      | Entry point used by Celery tasks. Loads TEI XML from disk and invokes the extractor.          |
| `extract_to_db.py`    | Writes parsed preprint metadata + reference rows into DynamoDB via `PreprintsRepo`.           |

Workflow:
1. `run_extract.extract_for_osf_id(provider_id, osf_id, base_dir)` locates `/data/preprints/<provider>/<osf_id>/tei.xml`.
2. Uses the shared `TEIExtractor` to parse title/doi/authors + reference items.
3. `write_extraction()` persists TEI summary + references and marks `preprints.tei_extracted`.

Celery task `osf_sync.tasks.enqueue_extraction` queues these jobs using the `by_queue_extract` GSI.

---

## Reference enrichment

| File                       | Description                                                                                   |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| `matching_crossref.py`     | Scores Crossref results to fill missing DOIs. Uses repo methods for selection + conditional updates. |
| `doi_check_openalex.py`    | Multi-stage OpenAlex lookup with fuzzy matching + threshold control.                          |
| `enrich_doi.py`            | Legacy helper combining Crossref/OpenAlex logic; uses the same repo update helpers.          |

Key functions:

- `matching_crossref.enrich_missing_with_crossref(limit, threshold, ua_email, ...)`
- `doi_check_openalex.enrich_missing_with_openalex(limit, threshold, mailto, osf_id, debug)`
- `enrich_doi.enrich_missing_with_crossref()` / `enrich_missing_with_openalex()` (thin wrappers around the modules above).

All enrichment functions:
1. Call `repo.select_refs_missing_doi(...)` to fetch work.
2. Query the respective API.
3. Use `repo.update_reference_doi(osf_id, ref_id, doi, source=...)` for conditional updates.

Celery tasks (`osf_sync.tasks.enrich_crossref` / `enrich_openalex`) pass parameters (limit, threshold, mailto) down to these functions.

---

## Running manually

Inside the container:

```bash
# Crossref enrichment
python -m osf_sync.augmentation.matching_crossref --limit 200 --threshold 78

# OpenAlex enrichment (debug mode, specific OSF id)
python -m osf_sync.augmentation.doi_check_openalex --osf_id <OSF_ID> --limit 50 --threshold 70 --debug

# Single TEI extraction (helpful for debugging)
python -m osf_sync.augmentation.run_extract --osf_id <OSF_ID> --provider-id <PROVIDER> --base /data/preprints
```

> The CLI arguments map directly to the `argparse` definitions at the bottom of each module.

---

## Dependencies

- `requests`, `thefuzz`, and TEI extractor utilities for parsing.
- `PreprintsRepo` for DynamoDB reads/writes (no direct SQLAlchemy usage).
- Rate-limited sleeps are built-in to avoid hitting Crossref/OpenAlex quotas.

---

## Tips

- Use `python -m osf_sync.dump_ddb --table preprint_references --limit 10` before/after enrichment to verify updates.
- When testing parsing, keep `PDF_DEST_ROOT` mounted locally so TEI files are accessible to `run_extract.py`.

