#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from osf_sync.db import init_db
from osf_sync.dynamo.client import get_dynamo_resource
from boto3.dynamodb.conditions import Key


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _scan_references(table) -> Dict[str, Dict[str, int]]:
    projection = "#oid, doi, has_doi"
    expr_names = {"#oid": "osf_id"}

    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "with_doi": 0})
    last_key: Optional[Dict[str, Any]] = None

    while True:
        kwargs = {"ProjectionExpression": projection, "ExpressionAttributeNames": expr_names}
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = table.scan(**kwargs)
        for item in resp.get("Items", []):
            osf_id = item.get("osf_id")
            if not osf_id:
                continue
            stats[osf_id]["total"] += 1
            doi = (item.get("doi") or "").strip()
            has_doi = _boolish(item.get("has_doi"))
            if has_doi and doi:
                stats[osf_id]["with_doi"] += 1
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
    return stats


def _fetch_refs_for_osf(table, osf_id: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    last_key: Optional[Dict[str, Any]] = None
    while True:
        kwargs = {"KeyConditionExpression": Key("osf_id").eq(osf_id)}
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = table.query(**kwargs)
        items.extend(resp.get("Items", []))
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
    return items

def main() -> None:
    parser = argparse.ArgumentParser(
        description="List OSF IDs where <threshold%% of references have DOIs."
    )
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum DOI ratio (default 0.2 = 20%%).")
    parser.add_argument("--min-refs", type=int, default=5, help="Only report OSF IDs with at least this many refs.")
    parser.add_argument("--table", default=None, help="References table name override.")
    parser.add_argument("--output", default=None, help="Optional JSONL output file.")
    parser.add_argument(
        "--dump-refs-dir",
        default=None,
        help="If set, dumps full reference lists for each low-coverage OSF ID into this directory (JSONL files).",
    )
    args = parser.parse_args()

    load_dotenv()
    init_db()

    table_name = args.table or os.getenv("DDB_TABLE_REFERENCES", "preprint_references")
    ddb = get_dynamo_resource()
    table = ddb.Table(table_name)
    stats = _scan_references(table)

    records = []
    for osf_id, counts in stats.items():
        total = counts["total"]
        with_doi = counts["with_doi"]
        if total < args.min_refs:
            continue
        ratio = with_doi / total if total else 0.0
        if ratio < args.threshold:
            records.append({"osf_id": osf_id, "total_refs": total, "with_doi": with_doi, "ratio": ratio})

    records.sort(key=lambda r: r["ratio"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    else:
        for rec in records:
            print(json.dumps(rec, ensure_ascii=False, default=str))

    if args.dump_refs_dir:
        dump_dir = Path(args.dump_refs_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        for rec in records:
            refs = _fetch_refs_for_osf(table, rec["osf_id"])
            safe_name = rec["osf_id"].replace("/", "_")
            dest = dump_dir / f"{safe_name}.jsonl"
            with open(dest, "w", encoding="utf-8") as fh:
                for ref in refs:
                    fh.write(json.dumps(ref, ensure_ascii=False, default=str) + "\n")
        print(f"Reference lists written to {dump_dir}")

    print(f"{len(records)} OSF IDs below {args.threshold*100:.1f}% DOI coverage (min refs {args.min_refs}).")


if __name__ == "__main__":
    main()
