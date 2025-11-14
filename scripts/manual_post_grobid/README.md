# Manual post-GROBID scripts

Run the downstream steps (GROBID, TEI extraction, enrichment) directly from your local Python environment — no Docker containers or Celery workers required. These scripts load `.env`, ensure DynamoDB tables exist, and use the same repo helpers as the main pipeline.

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
| `run_enrich_crossref.py`                 | Fills missing DOIs via Crossref.                                                  |
| `run_enrich_openalex.py`                 | Fills missing DOIs via OpenAlex.                                                  |
| `analyze_doi_sources.py`                 | Summarises DOI coverage (`by_source`, missing counts).                            |
| `dump_missing_doi_refs.py`               | Dumps remaining references without DOIs (JSON lines).                             |

Each script supports `--limit` (and optional `--dry-run`/`--sleep`) arguments; run with `-h` for details.

---

## Examples

```bash
# Parse 100 TEI files and write references
python scripts/manual_post_grobid/run_extraction.py --limit 100

# Crossref enrichment with stricter threshold
python scripts/manual_post_grobid/run_enrich_crossref.py --limit 300 --threshold 80

# OpenAlex enrichment with custom contact email
python scripts/manual_post_grobid/run_enrich_openalex.py --limit 200 --mailto you@example.com

# DOI coverage stats
python scripts/manual_post_grobid/analyze_doi_sources.py

# Dump missing DOI references
python scripts/manual_post_grobid/dump_missing_doi_refs.py --output missing.jsonl --limit 1000
```

These mirror the Docker/Celery tasks but execute sequentially on your local machine for quick experiments or ad-hoc runs.
