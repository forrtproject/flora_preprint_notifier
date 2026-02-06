# osf_sync.augmentation

Helpers for TEI extraction and reference enrichment.

## TEI extraction

| File | Purpose |
| --- | --- |
| `run_extract.py` | Parse TEI XML for one preprint and return extraction summary. |
| `extract_to_db.py` | Persist parsed TEI + references to DynamoDB via `PreprintsRepo`. |

Flow:
1. `run_extract.extract_for_osf_id(provider_id, osf_id, base_dir)` loads `/data/preprints/<provider>/<osf_id>/tei.xml`.
2. TEI parser extracts preprint metadata and references.
3. `write_extraction()` persists results and marks `preprints.tei_extracted`.

The pipeline stage for this is:
```bash
python -m osf_sync.pipeline run --stage extract --limit 200
```

## DOI enrichment

| File | Description |
| --- | --- |
| `matching_crossref.py` | Crossref scoring pipeline. |
| `doi_check_openalex.py` | OpenAlex lookup/fuzzy matching. |
| `doi_multi_method.py` | Combined DOI enrichment strategy used in the pipeline stage. |
| `forrt_original_lookup.py` | FORRT lookup + cache persistence. |
| `forrt_screening.py` | FORRT lookup/screen orchestration. |

Pipeline commands:
```bash
python -m osf_sync.pipeline run --stage enrich --limit 300
python -m osf_sync.pipeline run --stage forrt --limit-lookup 200 --limit-screen 500
```

## Manual commands

```bash
python -m osf_sync.augmentation.matching_crossref --limit 200 --threshold 78
python -m osf_sync.augmentation.doi_check_openalex --osf_id <OSF_ID> --limit 50 --threshold 70 --debug
python -m osf_sync.augmentation.forrt_screening --limit-lookup 200 --limit 500
python -m osf_sync.augmentation.run_extract --osf_id <OSF_ID> --provider-id <PROVIDER> --base /data/preprints
```

## Notes

- All writes happen through `PreprintsRepo`.
- FORRT payloads are cached in `api_cache` with TTL.
