#!/usr/bin/env python3
import argparse
from dotenv import load_dotenv

from osf_sync.extraction.extract_author_list import run_author_extract

load_dotenv()


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Run author extraction directly (no Celery queue).")
    ap.add_argument("--osf-id", action="append", dest="osf_ids", default=[])
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pdf-root", default=None)
    ap.add_argument("--keep-files", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-log", default=None)
    ap.add_argument("--match-emails-file", default=None)
    ap.add_argument("--match-emails-threshold", type=float, default=0.90)
    ap.add_argument("--include-existing", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    return run_author_extract(
        osf_ids=args.osf_ids or None,
        ids_file=args.ids_file,
        limit=args.limit,
        out=args.out,
        pdf_root=args.pdf_root,
        keep_files=args.keep_files,
        debug=args.debug,
        debug_log=args.debug_log,
        match_emails_file=args.match_emails_file,
        match_emails_threshold=args.match_emails_threshold,
        include_existing=args.include_existing,
    )


if __name__ == "__main__":
    raise SystemExit(main())
