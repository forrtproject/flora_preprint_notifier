# Manual post-GROBID scripts

Plain English: these are run-by-hand scripts for when you do not want Docker or Celery.

Run the downstream steps (GROBID, TEI extraction, enrichment) directly from your local Python environment - no Docker containers or Celery workers required. These scripts load `.env`, ensure DynamoDB tables exist, and use the same repo helpers as the main pipeline.

---

## Prerequisites

- Python environment with project dependencies installed (`pip install -r requirements.txt`).
- `.env` configured (local Dynamo or AWS).
- TEI/PDF files available under `PDF_DEST_ROOT` for extraction scripts.

Run scripts from the repository root so Python can import `osf_sync`:

```bash
# In PowerShell (Windows)
cd H:\fred_preprint_bot
$env:PYTHONPATH = "$PWD"
python scripts\manual_post_grobid\run_extraction.py --limit 25

# In bash/zsh (macOS/Linux)
cd /path/to/fred_preprint_bot
export PYTHONPATH=$(pwd)
python scripts/manual_post_grobid/run_extraction.py --limit 25
```

---

## Available scripts

| Script                                   | Description                                                                       |
| ---------------------------------------- | --------------------------------------------------------------------------------- |
| `run_extraction.py`                      | Parses TEI XML from disk and writes TEI/references back to DynamoDB.              |
| `doi_multi_method_lookup.py`             | Multi-method DOI matching and CSV output. Does not update DynamoDB.               |
| `run_forrt_screening.py`                 | FORRT lookup + screening to flag replications.                                    |
| `analyze_doi_sources.py`                 | Summarizes DOI coverage (`by_source`, missing counts).                            |
| `dump_missing_doi_refs.py`               | Dumps remaining references without DOIs (JSON lines).                             |
| `select_low_doi_coverage.py`             | Lists OSF IDs whose reference sets have <X% DOI coverage (optional ref dumps).    |
| `enqueue_author_extract.py`              | Enqueue author extraction tasks (optional helper).                                |

Each script supports `--limit` (and optional `--dry-run`/`--sleep`) arguments; run with `-h` for details.

---

## Examples

```bash
# Parse 100 TEI files and write references
python scripts/manual_post_grobid/run_extraction.py --limit 100

# Multi-method DOI matching (writes CSV only)
python scripts/manual_post_grobid/doi_multi_method_lookup.py --from-db --limit 200 --output doi_multi_method.csv

# FORRT lookup + screening
python scripts/manual_post_grobid/run_forrt_screening.py --limit-lookup 200 --limit 500

# DOI coverage stats
python scripts/manual_post_grobid/analyze_doi_sources.py

# Dump missing DOI references
python scripts/manual_post_grobid/dump_missing_doi_refs.py --output missing.jsonl --limit 1000

# OSF IDs with <20% DOI coverage and at least 30 refs (plus reference dumps)
python scripts/manual_post_grobid/select_low_doi_coverage.py --threshold 0.2 --min-refs 30 --dump-refs-dir low_refs
```

Note: `doi_multi_method_lookup.py` writes a CSV only. To update DynamoDB DOIs, use the `enrich-references` Celery task.

These mirror the Docker/Celery tasks but execute sequentially on your local machine for quick experiments or ad-hoc runs.
