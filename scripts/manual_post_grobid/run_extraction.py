#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv

from osf_sync.db import init_db
from osf_sync.dynamo.preprints_repo import PreprintsRepo
from osf_sync.augmentation.run_extract import extract_for_osf_id
from osf_sync.logging_setup import get_logger


log = get_logger("scripts.run_extraction")


def _load_env() -> None:
    load_dotenv()
    init_db()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse TEI XML and write references without Docker.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of TEI jobs to process.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Only list pending TEI jobs.")
    parser.add_argument("--base-dir", default=None, help="Override PDF/TEI base directory (defaults to PDF_DEST_ROOT).")
    args = parser.parse_args()

    _load_env()
    repo = PreprintsRepo()
    items = repo.select_for_extraction(args.limit)
    log.info("Selected TEI extractions", extra={"count": len(items)})

    if not items:
        return

    base_dir = args.base_dir or os.environ.get("PDF_DEST_ROOT", "/data/preprints")

    for idx, item in enumerate(items, start=1):
        osf_id = item.get("osf_id")
        provider_id = item.get("provider_id") or "unknown"
        if args.dry_run:
            log.info("DRY-RUN extraction", extra={"osf_id": osf_id, "provider_id": provider_id})
        else:
            summary = extract_for_osf_id(provider_id, osf_id, base_dir, raise_on_error=False)
            log.info("Extraction complete", extra={"osf_id": osf_id, **summary})
        if args.sleep and idx < len(items):
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()

