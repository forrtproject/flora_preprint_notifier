#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from osf_sync.db import init_db
from osf_sync.augmentation.matching_crossref import enrich_missing_with_crossref
from osf_sync.logging_setup import get_logger


log = get_logger("scripts.enrich_crossref")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Crossref enrichment without Docker/Celery.")
    parser.add_argument("--limit", type=int, default=300, help="Max references to check.")
    parser.add_argument("--threshold", type=int, default=78, help="Match threshold (0-100).")
    parser.add_argument("--osf-id", default=None, help="Restrict to a single OSF ID.")
    parser.add_argument("--ref-id", default=None, help="Restrict to a single reference within an OSF ID.")
    parser.add_argument("--ua-email", default=None, help="Override mailto header used by Crossref.")
    args = parser.parse_args()

    load_dotenv()
    init_db()
    stats = enrich_missing_with_crossref(
        limit=args.limit,
        threshold=args.threshold,
        ua_email=args.ua_email,
        osf_id=args.osf_id,
        ref_id=args.ref_id,
        debug=False,
    )
    log.info("Crossref enrichment finished", extra=stats)


if __name__ == "__main__":
    main()

