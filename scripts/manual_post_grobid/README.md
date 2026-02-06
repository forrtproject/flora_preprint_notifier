# Manual post-GROBID scripts

Run downstream TEI/extraction/enrichment workflows directly from local Python (no queue workers).

## Prerequisites

- Dependencies installed (`pip install -r requirements.txt`)
- `.env` configured
- Run from repo root

```bash
export PYTHONPATH=$(pwd)
```

## Available scripts

| Script | Description |
| --- | --- |
| `run_extraction.py` | Parse TEI XML and write TEI/references to DynamoDB. |
| `doi_multi_method_lookup.py` | Multi-method DOI matching and CSV output (no DB updates). |
| `run_flora_screening.py` | FLoRA lookup + screening. |
| `analyze_doi_sources.py` | DOI source coverage summary. |
| `dump_missing_doi_refs.py` | Dump references still missing DOI. |
| `select_low_doi_coverage.py` | Identify OSF IDs with low DOI coverage. |
| `enqueue_author_extract.py` | Run author extraction directly. |

## Examples

```bash
python scripts/manual_post_grobid/run_extraction.py --limit 100
python scripts/manual_post_grobid/doi_multi_method_lookup.py --from-db --limit 200 --output doi_multi_method.csv
python scripts/manual_post_grobid/run_flora_screening.py --limit-lookup 200 --limit 500
python scripts/manual_post_grobid/analyze_doi_sources.py
python scripts/manual_post_grobid/dump_missing_doi_refs.py --output missing.jsonl --limit 1000
python scripts/manual_post_grobid/select_low_doi_coverage.py --threshold 0.2 --min-refs 30
python scripts/manual_post_grobid/enqueue_author_extract.py --limit 100
```

To update DynamoDB DOIs, use:
```bash
python -m osf_sync.pipeline run --stage enrich --limit 300
```
