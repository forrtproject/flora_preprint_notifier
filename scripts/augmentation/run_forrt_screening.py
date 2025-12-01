"""
Standalone runner for FORRT lookup + screening without the docker entrypoint.
It delegates to the main osf_sync augmentation module.
"""
from __future__ import annotations

import argparse
import json

from osf_sync.augmentation.forrt_screening import lookup_and_screen_forrt


def main() -> None:
    ap = argparse.ArgumentParser(description="Run FORRT lookup + screening (no docker wrapper).")
    ap.add_argument("--limit-lookup", type=int, default=200, help="How many refs to send to FORRT lookup")
    ap.add_argument("--limit-screen", type=int, default=500, help="How many refs to screen")
    ap.add_argument("--osf_id", default=None, help="Optional OSF id to scope work")
    ap.add_argument("--ref_id", default=None, help="Optional ref id to scope work")
    ap.add_argument("--cache-ttl-hours", type=int, default=None, help="Override FORRT cache TTL")
    ap.add_argument("--include-checked", action="store_true", help="Re-run lookup even if status exists")
    ap.add_argument("--no-persist", action="store_true", help="Do not write screening flags back to DB")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging in the underlying module")
    args = ap.parse_args()

    out = lookup_and_screen_forrt(
        limit_lookup=args.limit_lookup,
        limit_screen=args.limit_screen,
        osf_id=args.osf_id,
        ref_id=args.ref_id,
        cache_ttl_hours=args.cache_ttl_hours,
        persist_flags=not args.no_persist,
        only_unchecked=not args.include_checked,
        debug=args.debug,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
